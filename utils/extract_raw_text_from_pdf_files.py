import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
SYSTEM_PROMPT = """You are a helpful assistant.  
You are proficient in extracting text from PDFS, using OCR when necessary.  
The text you extract must be returned in a markdown format with a header at the beginning of each page.  
The header includes the page number.  Format your response for optimal use for a
 language model RAG question/answer application.
 """
RAW_PDF_DIRECTORY = "../raw_pdf_files"
TEXT_FILE_DIRECTORY = "../text"
MODEL = "gemini-2.5-flash-preview-04-17"


def extract():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    os.makedirs(TEXT_FILE_DIRECTORY, exist_ok=True)

    for filename in os.listdir(RAW_PDF_DIRECTORY):

        pdf_file_path = os.path.join(RAW_PDF_DIRECTORY, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        output_file_path = os.path.join(TEXT_FILE_DIRECTORY, txt_filename)

        if os.path.exists(output_file_path):
            print(f"Skipping {filename} as {txt_filename} already exists.")
            continue

        print(f"Extracting text from {filename}...")
        files = [client.files.upload(file=pdf_file_path)]
        model = MODEL
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files[0].uri,
                        mime_type=files[0].mime_type,
                    ),
                    types.Part.from_text(text="Extract text from this PDF"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=SYSTEM_PROMPT),
            ],
        )
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Extracted text saved to {output_file_path}")


if __name__ == "__main__":
    extract()
