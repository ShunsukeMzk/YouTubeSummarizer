#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import librosa
import torch
import requests
import re
import shutil
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# ======== 設定 ========
# YouTube動画のURLをここに設定してください
YOUTUBE_URL = """
https://www.youtube.com/watch?v=YOUR_VIDEO_ID
""".strip()

# 出力ディレクトリ
OUTPUT_DIR = "./output"

# Whisper設定
STT_DEVICE = "mps" if torch.mps.is_available() else "cpu"
STT_MODEL_ID = "openai/whisper-large-v3-turbo"  # 文字起こしに使用するモデル

# LM Studio設定
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "openai/gpt-oss-120b"  # 要約に使用するモデル
# =====================


def ensure_dir(path: str) -> None:
    """ディレクトリが存在しない場合は作成"""
    os.makedirs(path, exist_ok=True)


def get_video_info(url: str) -> dict:
    """
    YouTube動画の情報を取得
    戻り値: 動画情報の辞書
    """
    try:
        with YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "不明なタイトル"),
                "uploader": info.get("uploader", "不明なアップローダー"),
                "upload_date": info.get("upload_date", ""),
                "duration": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "description": info.get("description", ""),
                "url": url,
            }
    except Exception as e:
        print(f"[WARNING] 動画情報の取得に失敗: {e}")
        return {
            "title": "不明なタイトル",
            "uploader": "不明なアップローダー",
            "upload_date": "",
            "duration": 0,
            "view_count": 0,
            "description": "",
            "url": url,
        }


def format_transcription(text: str) -> str:
    """
    文字起こしテキストを読点で改行して見やすくする
    戻り値: フォーマットされたテキスト
    """
    # 読点（。、！、？）の後に改行を挿入
    import re

    formatted_text = re.sub(r"([。！？])", r"\1\n", text)

    # 連続する改行を1つに統一
    formatted_text = re.sub(r"\n+", "\n", formatted_text)

    # 先頭と末尾の改行を削除
    formatted_text = formatted_text.strip()

    return formatted_text


def download_audio(url: str, output_dir: str) -> str:
    """
    YouTube動画から音声ファイルをダウンロード
    戻り値: ダウンロードされたファイルのパス
    """
    print(f"[INFO] 音声ダウンロード開始: {url}")

    # yt-dlpの設定
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "audio.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extractaudio": False,  # 音声抽出は後で行う
        "prefer_ffmpeg": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # 動画情報を取得
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "unknown_title")
            print(f"[INFO] 動画タイトル: {title}")

            # ダウンロード実行
            ydl.download([url])

            # ダウンロードされたファイルを探す
            for filename in os.listdir(output_dir):
                if filename.startswith("audio.") and not filename.endswith(".mp3"):
                    return os.path.join(output_dir, filename)

            raise FileNotFoundError("ダウンロードされたファイルが見つかりません")

    except DownloadError as e:
        print(f"[ERROR] ダウンロードに失敗: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}", file=sys.stderr)
        sys.exit(1)


def convert_to_mp3(input_file: str, output_dir: str) -> str:
    """
    音声ファイルをmp3に変換
    戻り値: 変換されたmp3ファイルのパス
    """
    print(f"[INFO] mp3変換開始: {input_file}")

    output_file = os.path.join(output_dir, "audio.mp3")

    try:
        # ffmpegコマンドを構築
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-ar",
            "44100",
            "-y",  # 既存ファイルを上書き
            output_file,
        ]

        # ffmpegを実行
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"[INFO] mp3変換完了: {output_file}")

        # 元のファイルを削除
        os.remove(input_file)
        print(f"[INFO] 元ファイルを削除: {input_file}")

        return output_file

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg変換に失敗: {e}", file=sys.stderr)
        print(f"[ERROR] エラー出力: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            f"[ERROR] ffmpegが見つかりません。ffmpegをインストールしてください。",
            file=sys.stderr,
        )
        sys.exit(1)


def transcribe_audio(mp3_file: str) -> str:
    """
    mp3ファイルを一括で文字起こし
    戻り値: 文字起こし結果のテキスト
    """
    print(f"[INFO] 文字起こし開始: {mp3_file}")
    start_time = time.time()

    try:
        # デバイスとデータ型の設定
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # モデルとプロセッサの読み込み
        print("[INFO] Whisperモデルを読み込み中...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            STT_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(STT_DEVICE)

        processor = AutoProcessor.from_pretrained(STT_MODEL_ID)

        # 音声読み込み（Whisperは16kHzを期待）
        print("[INFO] 音声ファイルを読み込み中...")
        audio_array, sr = librosa.load(mp3_file, sr=16000)

        print(
            f"[INFO] 音声ファイルを一括で処理します（長さ: {len(audio_array)/sr:.2f}秒）"
        )

        # 特徴量に変換
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        ).to(STT_DEVICE, dtype=torch_dtype)

        # 文字起こし実行
        print("[INFO] 文字起こしを実行中...")
        generated_ids = model.generate(
            **inputs,
            return_timestamps=True,
            task="transcribe",
            # language="japanese",
        )

        # デコード
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        # 読点で改行して見やすくする
        transcription = format_transcription(transcription)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"[INFO] 文字起こし完了: {processing_time:.2f}秒")

        return transcription

    except Exception as e:
        print(f"[ERROR] 文字起こしに失敗: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def summarize_with_lm_studio(transcription: str, video_info: dict) -> str:
    """
    LM StudioのAPIを使って文字起こし結果をマークダウン形式でまとめる
    戻り値: マークダウン形式の要約
    """
    print("[INFO] LM Studioで要約を生成中...")

    # プロンプトの構築
    system_prompt = """あなたは優秀な要約作成者です。以下の文字起こしテキストを読み、内容を整理してマークダウン形式で要約してください。

要約の際は以下の点に注意してください：
1. 主要なポイントを箇条書きで整理する
2. 重要な情報やキーワードを強調する
3. 内容を論理的に構造化する
4. 読みやすいマークダウン形式にする
5. 日本語で出力する"""

    user_prompt = f"""以下の動画「{video_info['title']}」の文字起こしテキストを要約してください：

{transcription}

上記の内容をマークダウン形式で要約してください。"""

    # APIリクエストの構築
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 131072,
        "stream": False,
    }

    try:
        # LM StudioのAPIにリクエスト
        response = requests.post(
            LM_STUDIO_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )

        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            print("[INFO] 要約の生成が完了しました")

            # 動画情報とモデル名を要約の先頭に追加
            summary_with_info = (
                create_summary_header(video_info, LM_STUDIO_MODEL) + summary
            )
            return summary_with_info
        else:
            print(f"[WARNING] LM Studio APIエラー: {response.status_code}")
            print(f"[WARNING] エラー内容: {response.text}")
            return create_fallback_summary(transcription, video_info)

    except requests.exceptions.ConnectionError:
        print(
            "[WARNING] LM Studioに接続できません。ローカルでLM Studioが起動しているか確認してください。"
        )
        return create_fallback_summary(transcription, video_info)
    except requests.exceptions.Timeout:
        print("[WARNING] LM Studio APIのタイムアウトが発生しました。")
        return create_fallback_summary(transcription, video_info)
    except Exception as e:
        print(f"[WARNING] LM Studio APIでエラーが発生しました: {e}")
        return create_fallback_summary(transcription, video_info)


def create_summary_header(video_info: dict, model_name: str) -> str:
    """
    要約のヘッダー部分を作成（動画情報、使用モデル、要約日時を含む）
    戻り値: マークダウン形式のヘッダー
    """
    # 現在の日時を取得
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    # アップロード日時のフォーマット
    upload_date = video_info.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        # YYYYMMDD形式をYYYY年MM月DD日に変換
        formatted_upload_date = (
            f"{upload_date[:4]}年{upload_date[4:6]}月{upload_date[6:8]}日"
        )
    else:
        formatted_upload_date = "不明"

    # 動画の長さを分:秒形式に変換
    duration = video_info.get("duration", 0)
    if duration > 0:
        minutes = duration // 60
        seconds = duration % 60
        duration_str = f"{minutes}分{seconds}秒"
    else:
        duration_str = "不明"

    # 再生回数をフォーマット
    view_count = video_info.get("view_count", 0)
    if view_count > 0:
        if view_count >= 1000000:
            view_str = f"{view_count // 1000000}M回"
        elif view_count >= 1000:
            view_str = f"{view_count // 1000}K回"
        else:
            view_str = f"{view_count}回"
    else:
        view_str = "不明"

    header = f"""# {video_info['title']} - 要約

## 動画情報

- **タイトル**: {video_info['title']}
- **アップローダー**: {video_info['uploader']}
- **アップロード日時**: {formatted_upload_date}
- **動画の長さ**: {duration_str}
- **再生回数**: {view_str}
- **URL**: {video_info['url']}

## 使用モデル

- **要約生成**: {model_name}

## 要約実施日時

- **要約作成日時**: {current_time}

---

"""

    return header


def create_fallback_summary(transcription: str, video_info: dict) -> str:
    """
    LM Studioが使用できない場合のフォールバック要約
    戻り値: 基本的なマークダウン形式の要約
    """
    print("[INFO] フォールバック要約を作成中...")

    # ヘッダーを作成
    header = create_summary_header(
        video_info, "フォールバック要約（LM Studio APIが利用できませんでした）"
    )

    # 基本的な要約を作成
    lines = transcription.split("\n")
    summary_lines = [
        "## 文字起こし結果",
        "",
        "### 内容",
        "",
    ]

    # 最初の10行程度を要約として使用
    for i, line in enumerate(lines[:10]):
        if line.strip():
            summary_lines.append(f"- {line.strip()}")

    if len(lines) > 10:
        summary_lines.append("")
        summary_lines.append(f"... 他 {len(lines) - 10} 行")

    summary_lines.extend(
        [
            "",
            "### 注意",
            "この要約は自動生成されたものです。詳細は文字起こし結果をご確認ください。",
            "",
            "---",
            "",
            "## 完全な文字起こし結果",
            "",
            transcription,
        ]
    )

    return header + "\n".join(summary_lines)


def sanitize_filename(filename: str) -> str:
    """
    ファイル名やディレクトリ名に使用できない文字を置き換える
    戻り値: 安全なファイル名
    """
    # WindowsとUnixで使用できない文字を置き換え
    # スペース、スラッシュ、バックスラッシュ、コロン、アスタリスク、クエスチョンマーク、ダブルクォート、パイプ、アングルブラケット
    invalid_chars = r'[<>:"/\\|?*]'
    filename = re.sub(invalid_chars, "_", filename)

    # 絵文字をアンダースコアに置き換え
    # 絵文字のUnicode範囲をカバー
    emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F0F5\U0001F200-\U0001F2FF]"
    filename = re.sub(emoji_pattern, "_", filename)

    # スペースをアンダースコアに置き換え
    filename = filename.replace(" ", "_")

    # 連続するアンダースコアを1つに統一
    filename = re.sub(r"_+", "_", filename)

    # 先頭と末尾のアンダースコアを削除
    filename = filename.strip("_")

    # 空文字列の場合はデフォルト名を使用
    if not filename:
        filename = "video"

    return filename


def rename_directory_to_title(old_dir: str, new_title: str) -> str:
    """
    ディレクトリ名を動画のタイトルに変更し、summarizedディレクトリに移動する
    戻り値: 新しいディレクトリのパス
    """
    if not os.path.exists(old_dir):
        print(f"[WARNING] ディレクトリが存在しません: {old_dir}")
        return old_dir

    # summarizedディレクトリを作成
    project_root = os.path.dirname(os.path.abspath(old_dir))
    summarized_dir = os.path.join(project_root, "summarized")
    ensure_dir(summarized_dir)

    # 安全なディレクトリ名を生成
    safe_title = sanitize_filename(new_title)

    # 新しいディレクトリのパス（summarizedディレクトリ内）
    new_dir = os.path.join(summarized_dir, safe_title)

    # 同名のディレクトリが既に存在する場合は番号を付ける
    counter = 1
    original_new_dir = new_dir
    while os.path.exists(new_dir):
        new_dir = f"{original_new_dir}_{counter}"
        counter += 1

    try:
        # ディレクトリを移動
        shutil.move(old_dir, new_dir)
        print(
            f"[INFO] ディレクトリを移動しました: {os.path.basename(old_dir)} → {os.path.basename(new_dir)}"
        )
        print(f"[INFO] 移動先: {new_dir}")
        return new_dir
    except Exception as e:
        print(f"[WARNING] ディレクトリの移動に失敗しました: {e}")
        return old_dir


def main():
    """メイン処理"""
    print("=== YouTube動画文字起こしスクリプト ===")

    # YouTubeのURLが設定されているかチェック
    if YOUTUBE_URL == "https://www.youtube.com/watch?v=YOUR_VIDEO_ID":
        print("[ERROR] スクリプト内のYOUTUBE_URLを設定してください。", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリを作成
    ensure_dir(OUTPUT_DIR)

    # 動画タイトルを保存する変数
    video_title = "動画"

    try:
        # 1. 音声ダウンロード
        audio_file = download_audio(YOUTUBE_URL, OUTPUT_DIR)

        # 2. mp3に変換
        mp3_file = convert_to_mp3(audio_file, OUTPUT_DIR)

        # 3. 文字起こし
        transcription = transcribe_audio(mp3_file)

        # 4. 結果を標準出力
        print("\n" + "=" * 50)
        print("文字起こし結果:")
        print("=" * 50)
        print(transcription)
        print("=" * 50)

        # 5. 動画情報を取得
        video_info = get_video_info(YOUTUBE_URL)
        video_title = video_info["title"]

        # 6. LM Studioで要約を生成
        summary = summarize_with_lm_studio(transcription, video_info)

        print("\n" + "=" * 50)
        print("要約結果:")
        print("=" * 50)
        print(summary)
        print("=" * 50)

        # 7. 結果をファイルに保存
        # 文字起こし結果
        transcription_file = os.path.join(OUTPUT_DIR, "transcription.txt")
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"\n[INFO] 文字起こし結果をファイルに保存しました: {transcription_file}")

        # 要約結果（マークダウン）
        summary_file = os.path.join(OUTPUT_DIR, "summary.md")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[INFO] 要約結果をマークダウンファイルに保存しました: {summary_file}")

        # 8. ディレクトリ名を動画のタイトルに変更
        print("\n[INFO] ディレクトリ名を動画のタイトルに変更中...")
        new_output_dir = rename_directory_to_title(OUTPUT_DIR, video_title)

        if new_output_dir != OUTPUT_DIR:
            print(f"[INFO] 新しいディレクトリ: {new_output_dir}")
            print(f"[INFO] すべての処理が完了し、ディレクトリ名を変更しました。")
        else:
            print(f"[INFO] すべての処理が完了しました。")

    except KeyboardInterrupt:
        print("\n[INFO] 処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
