import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
st.image(r"C:\Users\ravin\OneDrive\Pictures\Screenshots\Screenshot 2024-07-19 114337.png")
X = st.text_input("Inter your first word")
if st.button("predict"):
    Mdl1 =pickle.load(open('model1.pkl','rb'))
    Mdl2 =pickle.load(open('tk.pkl','rb'))


    for y in range(20):
        word = Mdl2.index_word[np.argmax(Mdl1.predict(pad_sequences(Mdl2.texts_to_sequences([X]),maxlen=23)))]
        X = X+" "+word
        st.write(X)
        time.sleep(0.9)