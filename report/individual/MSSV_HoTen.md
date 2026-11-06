# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604079
- **Họ và tên:** Hoàng Thị Thanh Nga
- **SEED:** 1684

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Model | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD/1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg | 0.899 | 0.889 | 0.14 | 0 |
| 2. BoW + MLP | **0.947** | 0.974 | 0.33 | 0 |
| 3a. Encoder + kNN | 0.858 | 0.974 | 79.19 | 0 |
| 3b. Encoder + LogReg | 0.934 | 1.000 | 66.50 | 0 |
| 3c. Encoder + MLP head | 0.939 | **1.000** | 40.86 | 0 |

*LLM không chạy được vì không có API key → các hàng 4a/4b/4c không có trong bảng.*

## 3. Dự đoán và thực tế

**3a. `hard_eval`** — dự đoán: BoW+LogReg 47%, BoW+MLP 45%, Enc+kNN 52%, Enc+LogReg 67%, Enc+MLP 68%; BoW chết ở `obfuscation` và `talks_about_spam`, LLM chết ở `prompt_injection`.

Thực tế: BoW+LogReg 54%, BoW+MLP 60%, **Enc+kNN 72%** (cao nhất), Enc+LogReg 62%, Enc+MLP 68%.

→ Lệch lớn ở chỗ mình tưởng Enc+kNN thấp nhất (52%) mà thực ra nó **cao nhất**. Lý do: kNN "dám đoán spam" nhiều hơn (recall 0.959 trên test thường), nên trên tập case khó (thiên về spam tinh vi) nó bắt được nhiều hơn; trong khi LogReg/MLP có precision cao → hay đoán ham → bỏ sót spam giọng tự nhiên.

**3b. `emb_tricky`** — dự đoán "Encoder+LogReg tốt hơn vì hiểu ngữ nghĩa". Thực tế: **cả hai đều đúng 5/6**, hòa. 6 câu tricky đều ngắn và có từ khoá tương đối rõ, BoW đủ để đoán đúng; khác biệt ngữ nghĩa không lộ ra ở bộ mẫu nhỏ.

**3c. `tradeoff`** — dự đoán "BoW, MiniLM+MLP trên Pareto; MiniLM+kNN nằm bên phải SLA 50ms". Thực tế: **BoW+LogReg, BoW+MLP, MiniLM+MLP head** trên Pareto; **cả 3 đạt SLA < 50ms** — 3c p95 = 40.86ms nên vẫn lọt, kNN (79ms) mới vượt.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất: `1. BoW + LogReg`** — F1 từ 0.889 xuống **0.378** (rơi 0.511 điểm). Lý do: precision = 1.0 trên test thường → thà bỏ sót còn hơn chặn nhầm → sang case khó gặp "spam giọng tự nhiên" và "né từ khoá" thì bỏ sót gần hết.

**Nhóm khó nhất theo từng model:**

- **BoW (cả LogReg và MLP):** `Né từ khoá` — 17% và **0%**; `Spam giọng tự nhiên` — 20%. Token kiểu `C0ngr4tulat10ns`, `F R E E` không khớp vocab → vector toàn 0 → đoán ham hết.
- **Embedding (3a/3b/3c):** `Spam giọng tự nhiên` — 40–50%; `Nói về spam` — 50%. Hiểu nghĩa tốt nhưng spam giọng bạn bè vẫn lẫn với ham thật.

**2 case cụ thể khiến model sai:**

- **#12** *"hi there, sorry to bother you, my friend gave me your number… meetlocal.club"* — **cả 5 model đều sai**. Bẫy: mở đầu lịch sự, không từ khoá. BoW: toàn từ ham ("friend", "chat") → đoán ham. Embedding: câu này *thật sự* giống tin làm quen, chỉ khác mỗi link `.club` — encoder không có tín hiệu "domain lạ".
- **#16** *"Long time no see! I've started a little side business with skincare…"* — **cả 5 model đều sai**. BoW: toàn từ ham. Embedding: giống y tin bạn cũ kể chuyện, không có tín hiệu MLM. Đây là **điểm mù chung**: spam giọng tự nhiên chỉ phân biệt được bằng **ngữ cảnh ngoài văn bản** — thứ mà cả 3 cách tiếp cận đều không có.

## 5. Thử thách cá nhân

**Đề #5 (theo SEED=1684):** Embedding — viết 4 tin (2 spam né từ khoá, 2 ham chứa "free"/"win"), so BoW vs Encoder+LogReg.

**Cách làm:** Viết 4 câu thủ công, chạy qua cả 2 pipeline đã build sẵn, in dự đoán từng câu.

**Kết quả:** BoW **2/4** · Encoder+LogReg **2/4** — cả hai đều tạch cả 2 case spam, đúng cả 2 case ham.

- **Case 1 (parcel £1.85 + link):** cả hai đoán ham, sai. BoW thấy "parcel/fee/delivery" toàn từ ham. Encoder thấy giống y tin giao hàng thật, không phân biệt được link giả.
- **Case 2 (bạn cũ bán wellness kit):** cả hai sai. Giống #16 trong bộ 50 case — **điểm mù "spam giọng tự nhiên"** mà thực nghiệm cho thấy Emb chỉ đạt 40%.
- **Case 3 ("free to grab coffee?") và Case 4 ("won the pub quiz… Free drinks"):** cả hai đúng. BoW đúng nhờ câu ngắn + các từ "coffee/pub quiz/Friday" đủ nặng để kéo "free/won" về phía ham.

**Bài học:** Embedding hiểu nghĩa tốt nhưng **không thay thế được classifier chuyên biệt** cho các dấu hiệu tinh vi (domain lạ, cấu trúc lừa đảo). Muốn bắt spam giọng tự nhiên phải có thêm feature ngoài văn bản (metadata người gửi, tần suất).

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (~116 tin/s trung bình), p95 < 50ms/tin.

**Đạt SLA:** chỉ có BoW+LogReg (0.14ms), BoW+MLP (0.33ms), MiniLM+MLP head (40.86ms).

**Chọn: `2. BoW + MLP`** — F1 0.947 (cao nhất trong các model đạt SLA), p95 0.33ms (dư sức), train 17s, **miễn phí**, throughput 8.892 tin/s.

**Không chọn LLM:** 10 triệu tin/ngày × (100 input + 1 output) = 1 tỷ input token + 10 triệu output token/ngày. Giá gpt-6-luna: $0.10/$0.50 mỗi 1M → **$100 + $5 = $105/ngày ≈ $3.150/tháng**. Chưa kể p95 của LLM ~ 2–3 giây/tin, **không đạt SLA 50ms** (chênh ~50 lần).

**Đánh đổi:** BoW+MLP rất nhanh, rẻ, nhưng sẽ yếu ở "né từ khoá" và "spam giọng tự nhiên" (stress test — F1 chỉ 0.50 ở nhóm này). Muốn vá thì thêm tầng 2 (router) — xem phần phản ánh.

## 7. AI log và điều đã học

**AI log:**

| Phần | Hỏi AI gì | AI trả lời | Đã sửa/kiểm chứng |
|---|---|---|---|
| Toàn bài | Không dùng AI để viết code Phần 1–3 . | — | — |
| Phần 4 | Không gọi API vì không có key, các ô tự skip. | — | — |
| Thử thách #5 | Không dùng AI — tự viết 4 câu theo đề. | — | — |
| Báo cáo | Hỏi AI gợi ý cách diễn đạt phần 4 (phân tích case khó). | AI gợi ý phân theo nhóm + cơ chế. | Đọc lại bảng `by_cat`, chỉnh lại % thực tế cho khớp. |

**Điều hiểu ra hôm nay mà trước đó chưa hiểu:**

1. **Precision cao không đồng nghĩa "tốt".** BoW+LogReg precision 1.0 trên test thường tưởng ngon, sang 50 case khó tụt xuống F1 0.378 — vì precision cao = model cực kỳ "nhát", thà bỏ sót còn hơn chặn nhầm. Trên dữ liệu khó (nhiều spam tinh vi) thì chính sự nhát đó giết F1. Chọn model phải theo **mục tiêu sản phẩm** (thà chặn nhầm hay thà bỏ sót), không nhìn F1 test sạch.

2. **"Test thường" lạc quan vì cùng phân phối.** Cả 5 model đều 0.86–0.95 trên test, nhưng sang 50 case khó tụt còn 0.38–0.74. Điều này nhắc: nếu chỉ đánh giá trên tập chia ngẫu nhiên thì không bao giờ biết model có thật sự "hiểu" spam hay chỉ học thuộc từ khoá.

3. **Không có model nào bắt được "spam giọng tự nhiên".** Cả 5 model đều sai 4/4 case nhóm này (#12, #13, #16, #19). Đây là giới hạn **bản chất của việc chỉ nhìn văn bản** — muốn bắt phải có metadata (người gửi, tần suất, thời điểm) — thứ mà cả 4 cách tiếp cận trong lab đều không có.

