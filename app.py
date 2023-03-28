from flask import Flask, request, jsonify, render_template, json, redirect, url_for
from flask_mongoengine import MongoEngine
from datetime import datetime
from flask_wtf import FlaskForm
from pythainlp.tokenize import word_tokenize, Tokenizer
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
from pymongo import MongoClient
from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie
import re
import tensorflow as tf
import numpy as np
import pythainlp
import os

app = Flask(__name__)

DB_URI = (
    "mongodb+srv://admin:1234@cluster0.seb1foe.mongodb.net/?retryWrites=true&w=majority")
app.config['MONGODB_HOST'] = DB_URI
db = MongoEngine(app)


class Employee(db.Document):
    text = db.StringField()
    pub_date = db.DateTimeField(datetime.now)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dict', methods=['GET', 'POST'])
def query_records():
    employee = Employee.objects.all()
    return render_template('dict.html', employee=employee)


@app.route('/updateemployee', methods=['POST'])
def updateemployee():
    pk = request.form['pk']
    namepost = request.form['name']
    value = request.form['value']
    employee_rs = Employee.objects(id=pk).first()
    if not employee_rs:
        return json.dumps({'error': 'data not found'})
    else:
        if namepost == 'text':
            employee_rs.update(name=value)
    return json.dumps({'status': 'OK'})


@app.route('/add', methods=['GET', 'POST'])
def create_record():
    txtname = request.form['txtname']
    employee_r = Employee.objects.all()
    if(txtname == employee_r):
        t = "คำนี้มีเเล้ว"
        return render_template('index.html', t)
    else:
        employeesave = Employee(text=txtname)
        employeesave.save()
    return redirect('/dict')


@app.route('/delete/<string:getid>', methods=['POST', 'GET'])
def delete_employee(getid):
    print(getid)
    employeers = Employee.objects(id=getid).first()
    if not employeers:
        return jsonify({'error': 'data not found'})
    else:
        employeers.delete()
    return redirect('/dict')


model = tf.keras.models.load_model('model2.h5')
# สร้าง dict สำหรับแปลงค่าความน่าจะเป็นเป็นข้อความ
label_map = {
    0: 'ไม่เป็นคำหยาบ',
    1: 'คำหยาบ'
}


@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    text = request.form['text']
    if text:
        # ทำนาย
        label, probability, tokens = predict(text)
        # ส่งผลลัพธ์กลับไปหาผู้ใช้
        result = {
            'text': text,
            'label': label,
            'probability': probability,
            'tokens': tokens,
        }
    else:
        result = None
    return render_template('index.html', result=result)

def predict(text):
    # ตัดคำด้วย newmm
    tokens = pythainlp.tokenize.word_tokenize(text, engine='newmm')
    # แปลงคำเป็นตัวเลขโดยใช้ tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=21)
    # ทำนาย
    result = model.predict(data)
    label = np.argmax(result)
    probability = result[0][label]
    # แปลงค่าความน่าจะเป็นเป็นข้อความ
    label_text = label_map[label]
    # ตัวแปร token_string เพื่อเก็บ string ของ tokens
    token_string = '|'.join(tokens)
    # สร้าง list ของคำหยาบ
    employees = Employee.objects.all()
    # เช็คว่ามีคำหยาบใน text หรือไม่
    has_abusive_word = any(word in employee.text for word in tokens for employee in employees)
    # สร้าง label ขึ้นมาเพื่อระบุว่าพบคำหยาบหรือไม่
    if has_abusive_word:
        label_text = 'พบคำหยาบ'
    else:
        label_text = 'ไม่เป็นคำหยาบ'
    return label_text, probability, token_string
if __name__ == '__main__':
    app.run()