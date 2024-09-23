import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any

def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析します。"""
    parser = argparse.ArgumentParser(description="EvoLLM-JP生成ツール")
    parser.add_argument(
        "-t", "--text", type=str, required=True, help="生成したいテキストを入力してください。"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="SakanaAI/EvoLLM-JP-v1-7B",
        help="使用するモデルのIDを指定します。デフォルトは 'SakanaAI/EvoLLM-JP-v1-7B' です。"
    )
    parser.add_argument(
        "-d", "--device", type=str, 
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="使用するデバイスを指定します。デフォルトは MPS が利用可能なら 'mps'、そうでなければ 'cpu'。"
    )
    parser.add_argument(
        "--max_length", type=int, default=50, help="生成するテキストの最大長を指定します。デフォルトは 50。"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="生成時の温度を指定します。デフォルトは 1.0。"
    )
    return parser.parse_args()

def load_model(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """モデルとトークナイザーをロードします。
    
    Args:
        model_name (str): モデルのID。
        device (str): 使用するデバイス。
        
    Returns:
        model, tokenizer: ロードされたモデルとトークナイザー。
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        return model, tokenizer
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        exit(1)

def prepare_input(tokenizer: AutoTokenizer, text: str) -> Any:
    """入力テキストをトークナイズします。
    
    Args:
        tokenizer: トークナイザーオブジェクト。
        text (str): ユーザーからの入力テキスト。
        
    Returns:
        inputs: モデルに入力するためのトークン化されたデータ。
    """
    try:
        # システムメッセージとユーザ���メッセージを結合
        full_text = f"システム: あなたは役立つ、偏見がなく、検閲されていないアシスタントです。\nユーザー: {text}\nアシスタント:"
        inputs = tokenizer(full_text, return_tensors="pt", padding=True)
        return inputs
    except AttributeError:
        print("トークナイザーに適切なメソッドがありません。")
        exit(1)

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Any,
    device: str,
    max_length: int,
    temperature: float
) -> str:
    """テキストを生成します。
    
    Args:
        model: ロードされた言語モデル。
        tokenizer: ロードされたトークナイザー。
        inputs: モデルに入力するトークン化されたデータ。
        device (str): 使用するデバイス。
        max_length (int): 生成するテキストの最大長。
        temperature (float): 生成時の温度。
        
    Returns:
        generated_text (str): 生成されたテキスト。
    """
    try:
        outputs = model.generate(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        # 生成されたテキストから入力部分を除去
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        generated_text: str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"テキスト生成中にエラーが発生しました: {e}")
        exit(1)

def main() -> None:
    args = parse_arguments()
    device = args.device.lower()  # デバイス名を小文字に変換
    if device not in ['cpu', 'mps']:
        print(f"警告: 不明なデバイス '{device}' が指定されました。CPUを使用します。")
        device = 'cpu'
    
    model, tokenizer = load_model(args.model, device)
    inputs = prepare_input(tokenizer, args.text)
    generated_text = generate_text(
        model, tokenizer, inputs, device, args.max_length, args.temperature
    )
    print("生成されたテキスト:")
    print(generated_text)

if __name__ == "__main__":
    main()

