# pdf-gpt
A project that uses the GPT API to find what you need using a PDF file as context.

## requirements
The OPENAI_API_KEY environment variable is required on the execution machine.

## install
```bash
poetry install
```


## usage
See the file `use_pdf_gpt.ipynb`

Example screenshot using the `./documents/article.pdf` file
![USAGE EXAMPLE](./misc/screenshot.png)


## todo
- Tuning answer performance for long-form PDFs(+ Especially Korean text): Current methods don't produce good answer 
- Add a question method to remember previous conversations
- Adding a web UI