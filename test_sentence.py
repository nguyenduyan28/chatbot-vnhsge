from sentence_transformers import SentenceTransformer
import re
from sklearn.cluster import AgglomerativeClustering
from functools import reduce
from operator import concat

sen = """
Tỉ trọng của Nhật Bản trong nên sản xuất của thế giới là 1/10. GDP của Nhật Bản năm 2000 là 4 746 tỉ USD và bình quân GDP trên đẩu người là 37 408 USD. Khoa hoc kĩ thuật của Nhật Bản vẫn tiếp tục phát triển ở trình độ cao. Tính đến năm 1992, Nhật Bản đã phóng 49 vệ tinh khác nhau và hợp tác có hiệu quả với Mĩ, Liên Xô (sau là Liên bang Nga) , trong các chương trình vũ trụ quốc tế. Vê văn hoá, tuy là một nước tư bản phát triển cao, nhưng Nhật Bản vẫn giữ được những giá trị truyên thống và bản sắc văn hoá của mình. Sự kết hợp hài hoà giữa truyên thống và hiện đại là nét đáng chú ý trong đời sống văn hoá Nhât Bản. Vể chính trị, sau 38 năm Đảng Dân chủ Tư do liên tuc cầm quyền (1955 1993), từ năm 1993 đến năm 2000. chính quyên ở Nhật Bản thuộc về các đảng đối lập hoặc liên minh các đảng phái khác nhau , tình hình xã hội Nhật Bản có phẩn không ổn đinh. Trận động đất ở Côbê (1 1995) đã gây thiệt hại lớn vể người và của vụ khủng bố bẳng hơi độc trong đường tàu điện ngâm của giáo phái Aum (3 - 1995) và nạn thất nghiệp tăng cao v.v. đã làm cho nhiểu người dân Nhât Bản hết sức lo Iắng Về đối ngoại, Nhật Bảntiếp tục duy trì sựliên minh chặt chẽ với Mĩ. Tháng 4 - 1996, hai nước ra tuyên bố khẳng < định lại việc kéo dài vĩnh viễn Hiệp ước an ninh Mĩ ~ Nhật. Mặt khác, với học thuyết Miyadaoa (1 1993 ), và học thuyết Hasimôtô (1 - 1997), Nhật Bản vẫn coi trọng quan hệ với Tây Âu, mở rộng hoạt động đối ngoai với các đối tác khác đến phạm vi toàn cầu và chú trọng phát triển quan hệ với các nước Đông Nam Á Từ đẩu những năm 90, Nhật Bản nỗ lưc vươn lên thành một cường quốc chính trị để tương xứng với vịthế siêu cường kinh tế. Nêu những nét cơ bản vể tình hình kinh tế và chính trị của Nhật Bản trong thập kỉ 90 của thế kỉ XX CÂU HỎI VÀ BÀI TẬP Những yếu tố nào khiến Nhật Bản trở thành một trong ba trung tâm kinh tế tài chính của thế giới vào nửa cuối thế kỉ XX ? 2. Khái quát chính sách đối ngoại của Nhật Bản từ sau Chiến tranh thế giới thứ hai đến năm 2000. 57
"""
def chunk_from_topic(sentences):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    sentences = re.split(r'(?<=[.!?])\s+', sentences.strip())
    embeddings = model.encode(sentences)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2)
    labels = clustering.fit_predict(embeddings)

    sub_paragraphs = {}
    for sentence, label in zip(sentences, labels):
        sub_paragraphs.setdefault(label, []).append(sentence)
    chunk = []
    chunk += (s for s in sub_paragraphs.values())
    return list(reduce(concat, chunk))
    
    from sentence_transformers import SentenceTransformer, util

# Load Pre-Trained Model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def semantic_chunk(text, max_len=300):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Test the Chunking Function
