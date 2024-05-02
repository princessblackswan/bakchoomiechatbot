# bakchoomiechatbot
Upload your own documents (.pdf, .docx, or .txt) and chat with them!

What you need:
1. Your own OpenAI API Key
2. Any .pdf, .docx, or .txt documents of your choice!

How to adjust the chatbot parameters:
- Chunks: refers to the fixed no of words by which the documents are 'chunked' by. Do play around with the chunk sizes (max value 2048) to find your best setting. This would depend on the document you upload.
- k: This refers to the number of vectors fetched from your original document when you are asking a question to the chatbot. It is basically the number of nearest neighbors to retrieve or consider in a similarity search. If you want more comprehensive answer, increase your k. If you want short, precise answers, reduce your k. The max value of k is 20

To reproduce in your machine, pls install the library in the requirements.txt file.

Please note that if you have the latest version of Streamlit installed, you need to rollback to streamlit==1.24.0 to ensure it works

Have fun!
