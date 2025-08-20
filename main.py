# 導入所需的套件
import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
#from streamlit_extras.stylable_container import stylable_container  # 若未裝可刪掉本行與相關用法


# 網頁基本配置
st.set_page_config(page_title="中文客服檢索回覆", page_icon="♟️", layout="wide")
st.title("中文客服檢索回覆系統")

# ---- 你的既有程式 ----
DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"你們的營業時間是？","answer":"我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。"},
        {"question":"如何申請退貨？","answer":"請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。"},
        {"question":"運費如何計算？","answer":"單筆訂單滿 NT$ 1000 免運，未滿則酌收 NT$ 80。"},
        {"question":"可以開立發票嗎？","answer":"我們提供電子發票，請於結帳時填寫統一編號與抬頭。"},
    ]
)

if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None

st.subheader("上傳知識庫")
uploader = st.file_uploader("限上傳 CSV 檔案", type=["csv"])
if uploader is not None:
    df = pd.read_csv(uploader)
    st.session_state.faq_df = df.dropna().reset_index(drop=True)
    st.success(f"上傳成功 {len(df)} 筆資料！")

with st.expander("檢視資料", expanded=False):
    st.dataframe(st.session_state.faq_df, use_container_width=True)

do_index = st.button("建立/重設索引")

def jieba_tokenize(text: str):
    return list(jieba.cut(text))

if do_index or (st.session_state.vectorizer is None):
    corpus = (st.session_state.faq_df["question"].astype(str) + " " +
              st.session_state.faq_df["answer"].astype(str)).tolist()
    v = TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf = v.fit_transform(corpus)
    st.session_state.vectorizer = v
    st.session_state.tfidf = tfidf
    st.success("索引建立成功！")

q = st.text_input("請輸入您的問題(中文)", placeholder="例如：如何申請退貨？")

# 兩個原生 sliders（保持 aria-label 不變，當成定位錨點）
checks = st.columns(2)

with checks[0]:
    top_k = st.slider("顯示前幾筆相關結果", 1, 10, 3, key="top_k_slider")
with checks[1]:
    c= st.slider("信心門檻", 0.0, 1.0, 0.5, 0.05, key="c_slider")



# 當使用者按下送出按鈕時
if st.button("送出") and q.strip():
    if (st.session_state.tfidf is None) or (st.session_state.vectorizer is None):
        st.warning("尚未建立索引, 將自動建立索引！")
        corpus = (st.session_state.faq_df['question'].astype(str)+""+
                  st.session_state.faq_df['answer'].astype(str)).tolist()
        tfidf = v.fit_transform(corpus)
        st.session_state.vectorizer = v
        st.session_state.tfidf = tfidf
        st.success("索引建立成功！")

    # 計算相似度  
    vec= st.session_state.vectorizer.transform([q])
    sims = linear_kernel(vec, st.session_state.tfidf).flatten()
    idxc = sims.argsort()[::-1][:top_k]
    # 篩選符合信心門檻的結果
    rows = st.session_state.faq_df.iloc[idxc].copy()
    # 將相似度分數加入結果
    rows['score'] = sims[idxc]

    best_ans = None
    best_scores = float(rows['score'].iloc[0]) if len(rows) else 0.0
    if best_scores >= c:
        best_ans= rows['answer'].iloc[0]
        st.balloons()
    else:
        st.snow()

    if best_ans:
        st.success(f"找到相關答案: {best_ans}")
    else:
        st.info("找不到合適的答案，請聯繫客服。")

    # 展示可能的回答
    with st.expander("可能的回答", expanded=False):
        st.dataframe(rows[['question', 'answer', 'score']], use_container_width=True)