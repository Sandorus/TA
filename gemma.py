import os
import google.generativeai as genai  # <-- note: "generativeai", not "genai"

def main():
    # 1. Configure API key

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")


    genai.configure(api_key=GEMINI_API_KEY)

    # 2. Load Gemma 3 model (IT = Instruction Tuned)
    model = genai.GenerativeModel("gemma-3-27b-it")

    print("Enter your prompt (type 'exit' to quit):")
    while True:
        prompt = input("> ")
        if prompt.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = model.generate_content(f"{prompt}\n\nKeep your answer concise: no more than one short sentence. And Use Speech Synthesis Markup Language (SSML)",
        generation_config=genai.types.GenerationConfig(
        max_output_tokens=200
    )
)
        print("\nGemma 3 says:")
        print(response.text)
        print("-" * 40)

if __name__ == "__main__":
    main()
