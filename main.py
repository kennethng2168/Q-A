import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined Q&A pairs

qa_pairs = [
    {"question": "Bagaimana cara untuk membuat aduan?",
     "answer": "Cara membuat aduan:\n\nMelalui Biro Pengaduan Awam (BPA)\nMelalui Sistem Talian Aduan Rakyat Selangor (STARS)\nMelalui surat rasmi kepada jabatan\nMelalui emel JAIS; info@jais.gov.my\nMelalui talian telefon; 03-55143425 (Unit Perhubungan Awam & Penerbitan, JAIS)"},

    {"question": "Apakah yang dimaksudkan dengan kelakuan atau perbuatan tidak sopan di tempat awam?",
     "answer": "Kelakuan tidak sopan di tempat awam bermaksud apa-apa perbuatan yang dilakukan oleh mana-mana orang Islam dengan sengaja iaitu yang bertentangan dengan syarak secara terbuka di mana-mana tempat awam."},

    {"question": "Adakah menjadi satu kesalahan bagi seseorang yang beragama Islam menjual minuman keras?",
     "answer": "Ya. Perbuatan menjual minuman keras bagi orang Islam adalah salah menurut seksyen 18(2) Enakmen Jenayah Syariah (Negeri Selangor) tahun 1995. Sabit kesalahan boleh didenda tidak melebihi RM5000 atau dipenjarakan selama tempoh tidak melebihi tiga tahun atau kedua-duanya sekali. Selain itu, adalah menjadi satu kesalahan bagi seseorang yang beragama Islam membuat, menawarkan atau mempamerkan untuk jualan, menyimpan atau membeli apa-apa minuman yang memabukkan."},

    {"question": "Adakah pihak Penguatkuasa JAIS boleh mengambil tindakan bagi kes yang berlaku di kilang-kilang?",
     "answer": "Tindakan boleh dilakukan di mana-mana kawasan termasuk di kilang-kilang, jika ada maklumat awam yang lengkap dan berlakunya pelanggaran hukum syarak."},

    {
        "question": "Apakah tindakan pihak Penguatkuasa JAIS terhadap kegiatan maksiat yang berlaku di pusat membeli belah, taman rekreasi dan pusat hiburan terbuka secara terang-terangan?",
        "answer": "Bahagian Penguatkuasaan menjalankan aktiviti rondaan berjadual di kawasan-kawasan Hotspot (kawasan merah) bagi membendung kegiatan maksiat yang dibuat secara terang-terangan seperti tempat membeli-belah, taman rekreasi dan pusat hiburan terbuka setiap hari mengikut lokasi-lokasi sasaran yang dikenalpasti. Kegiatan berkelakuan tidak sopan di tempat-tempat awam boleh diambil tindakan di bawah seksyen 31 Enakmen Jenayah Syariah (Negeri Selangor) tahun 1995. Sabit kesalahan boleh dikenakan denda sebanyak RM1000 ataupun 6 bulan penjara atau kedua-duanya sekali."},

    {
        "question": "Adakah orang awam mempunyai kuasa menangkap? Siapakah individu lain yang mempunyai kuasa menangkap selain Pegawai Penguatkuasa Agama JAIS?",
        "answer": "Orang awam tidak mempunyai kuasa untuk menjalankan tangkapan. Walaubagaimanapun orang awam boleh membantu Pegawai Penguatkuasa Agama JAIS ataupun Pegawai Masjid untuk menjalankan tangkapan dengan menyalurkan maklumat. Selain Pegawai Penguatkuasa Agama JAIS, pegawai-pegawai masjid yang terdiri daripada Nazir, Imam 1, Bilal 1 dan Siak 1 turut diberi kuasa oleh Enakmen Pentadbiran Agama Islam untuk menjalankan tangkapan di qariah masing-masing."},

    {
        "question": "Siapakah yang boleh dihubungi untuk membuat tangkapan terhadap pasangan yang berkhalwat selain dari penguatkuasa JAIS?",
        "answer": "Pegawai Masjid yang terdiri daripada Nazir, Imam, Bilal dan Siak adalah orang boleh dihubungi untuk membuat tangkapan kes khalwat kerana mereka telah diberi kuasa oleh Majlis Agama Islam Selangor (MAIS). Selain itu, pengadu juga boleh terus memberi maklumat perlakuan jenayah Syariah ke mana-mana Balai Polis berhampiran."},

    {"question": "Bolehkah tindakan diambil dengan segera ke atas aduan yang dibuat?",
     "answer": "Tindakan penguatkuasaan akan diambil dengan segera ke atas aduan yang lengkap berstatus FFA (untuk tindakan lanjut) dalam tempoh 24 jam dan jika ketiadaan penguatkuasa agama disebabkan hal-hal tertentu yang tidak dapat dielakkan maka pegawai masjid atau pegawai polis boleh membantu untuk membuat tangkapan."},

    {"question": "Bagaimana untuk membuat aduan?",
     "answer": "Sebarang aduan boleh disalurkan melalui talian hotline 24 jam Bahagian Penguatkuasaan JAIS di talian 1800-88-2424 ataupun boleh hadir terus ke:\nHotline Penguatkuasaan JAIS,\nAras Bawah, Menara Utara,\nBangunan Sultan Idris Shah,\nSeksyen 5, Persiaran Masjid,\n40676 Shah Alam, Selangor."},

    {"question": "Apakah yang perlu saya lakukan sekiranya berlaku kejadian khalwat di kawasan saya?",
     "answer": "Apabila kejadian khalwat berlaku di kawasan anda langkah-langkah berikut perlu diikuti:\n\n1. Melaporkan kejadian tersebut melalui talian tetap hotline 24 jam: 1800-88-2424\n2. Memberikan maklumat diri kepada pegawai bertugas.\n3. Memberikan maklumat kejadian khalwat seperti alamat kejadian yang tepat, nama dan latar belakang orang yang disyaki (jika berkenaan) serta tanda-tanda kejadian disyaki berlaku seperti wujudnya kenderaan orang yang disyaki."},

    {"question": "Bolehkah pengadu mengetahui status aduan yang telah dibuat kepada pihak JAIS?",
     "answer": "Status aduan yang lengkap boleh disemak oleh pengadu secara lisan dalam tempoh 10 hari selepas membuat aduan."},

    {"question": "Mengapakah pemberi maklumat perlu mengemukakan maklumat diri kepada pihak JAIS?",
     "answer": "Maklumat pengadu diperlukan supaya boleh direkodkan dan dirujuk oleh pihak Penguatkuasa Agama pada bila-bila masa diperlukan serta menunjukkan kebertanggungjawaban pengadu terhadap aduan yang dikemukakan."},

    {
        "question": "Adakah pengadu akan dirahsiakan dan tidak akan dipanggil di mahkamah jika ada perbicaraan? Adakah identiti pengadu akan dirahsiakan dan dilindungi jika berlaku ancaman keselamatan?",
        "answer": "Pihak pengadu akan dirahsiakan identiti untuk memelihara keselamatan pengadu. Walaubagaimanapun, pengadu hendaklah tampil apabila diperlukan pada bila-bila masa sekiranya diperlukan bagi tujuan siasatan."},

    {"question": "Apakah risiko ke atas pengadu yang membuat aduan ke penguatkuasa JAIS?",
     "answer": "Seseorang yang membuat aduan atau memberikan apa-apa maklumat tentang suatu perlakuan jenayah syariah hendaklah memberikan secara benar dan jika didapati aduan yang diberikan itu palsu maka pengadu boleh disabitkan dengan suatu kesalahan menurut Seksyen 211 Enakmen Tatacara Jenayah Syariah Negeri Selangor No. 3 Tahun 2003."}
]



# Embed Q&A pairs into FAISS vector store
def get_vector_store(qa_pairs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    chunks = [f"{qa['question']} {qa['answer']}" for qa in qa_pairs]
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_qa_index")


# Handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    db = FAISS.load_local("faiss_qa_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    # Return the most relevant answer
    if docs:
        response_text = docs[0].page_content.split(" ", 1)[1]  # Extract the answer part
        return response_text
    else:
        return "Maaf, saya tidak dapat menemukan jawapan untuk soalan anda."


def main():
    st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
    st.title("Legal Q&A Chatbot")
    st.write("Tanya saya soalan mengenai undang-undang syariah.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tanya saya apa-apa soalan."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawapan..."):
                response = user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)


if __name__ == "__main__":
    # Initialize the vector store with Q&A pairs on first run
    if not os.path.exists("faiss_qa_index"):
        get_vector_store(qa_pairs)
    main()