import sys, json, re, urllib.request, pathlib

spec_path, out_path = sys.argv[1], sys.argv[2]
spec = pathlib.Path(spec_path).read_text(encoding="utf-8")
prompt = (
    "Ты — генератор кода. Сгенерируй ПОЛНЫЙ файл Python строго по спецификации ниже.\n"
    "Выведи ТОЛЬКО исходный код файла: без markdown-ограждений, без пояснений, без <think>, "
    "без текста до или после кода. Реализуй ВСЕ функции полностью, без заглушек.\n\n" + spec
)
req = {
    "model": "qwen3-coder:30b", "prompt": prompt, "stream": False,
    "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
}
data = json.dumps(req).encode("utf-8")
r = urllib.request.urlopen(
    urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data,
                           headers={"Content-Type": "application/json"}),
    timeout=900,
)
resp = json.loads(r.read().decode("utf-8"))
text = resp.get("response", "")
text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
if m:
    text = m.group(1).strip()
mk = text.find('"""Авто-сегментация')
if mk > 0:
    text = text[mk:]
code = text.strip() + "\n"
pathlib.Path(out_path).write_text(code, encoding="utf-8")
print("WROTE", out_path, len(code), "chars,", code.count(chr(10)) + 1, "lines")