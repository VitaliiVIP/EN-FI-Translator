from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fi")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fi")

@app.route('/')
def home():
    return render_template('HTML.html', translation=None, text_to_translate="")
@app.route('/translate', methods=['POST'])
def translate_text():
    for_translate = request.form['text_to_translate']
    pretext = f">>fi<< {for_translate}"
    translated = model.generate(**tokenizer([pretext], return_tensors="pt", padding=True))
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    if "."  "!" "?" in for_translate:
        return render_template('HTML.html', translation=translation, text_to_translate=for_translate)
    else:
        if "." in translation:
            translation=translation[:-1]
            return render_template('HTML.html', translation=translation, text_to_translate=for_translate)
        else:
            return render_template('HTML.html', translation=translation, text_to_translate=for_translate)


if __name__ == '__main__':
    app.run(debug=True)

