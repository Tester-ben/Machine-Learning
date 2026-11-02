# Nhận xét bài tập 1–6

## Bài 1 – Tự vẽ cây

Đã xuất hai hình cây năm lá và cây sâu 2. Hai lá bên trái của cây sâu 2 đều kết luận trễ hạn: lần lượt có [68, 53] và [114, 14] khách thuộc [lớp 0, lớp 1]. Cây vẫn chia vì tỷ lệ hai lớp khác nhau. Gini nút cha là 0,3933, Gini trung bình có trọng số sau chia còn 0,3394, giảm 0,0540. Cây tối ưu độ thuần, không bắt buộc hai lá có nhãn dự đoán khác nhau.

## Bài 2 – Tự tìm câu hỏi gốc

Hàm `tim_cau_hoi(X, y)` duyệt từng cột, lấy các giá trị phân biệt đã sắp xếp, rồi thử trung điểm giữa hai giá trị kề nhau. Các ngưỡng nằm trong cùng một khoảng tạo ra cùng một cách chia, nên không cần thử mọi số thực.

Với 450 khách của tập học, câu hỏi tốt nhất là `thu_nhap <= 15.05`: nhánh trái 249 khách, nhánh phải 201 khách. Gini gốc 0,499842; Gini sau chia 0,377697; mức giảm 0,122145. Kết quả trùng nút gốc của cây Gini trong scikit-learn trên cùng tập học.

Gini sau chia phải được lấy trung bình có trọng số theo số khách ở mỗi nhánh. Vì Gini gốc cố định, tìm mức giảm lớn nhất tương đương tìm Gini sau chia nhỏ nhất. Không dùng 150 khách kiểm tra để chọn ngưỡng. Kết luận về nút gốc không áp dụng cho mọi cây thành viên của rừng, vì các cây đó được bốc mẫu và chọn cột ngẫu nhiên.

## Bài 3 – Gini và entropy


Entropy thấp hơn khoảng 0,67 điểm phần trăm, tương ứng một khách trong 150 khách kiểm tra. Đổi thước đo không làm kết quả thay đổi nhiều trong lần chia này. Độ sâu được chọn bằng CV trên tập học; khi bằng điểm, chọn giá trị xuất hiện trước trong danh sách tăng dần, giống cách chọn của F3.

## Bài 4 – Số khách tối thiểu ở lá



Chọn 10 khi 10 và 20 đồng điểm. Cây không giới hạn độ sâu này có độ sâu 8, 30 lá, accuracy kiểm tra 0,8467. Cây giới hạn độ sâu 5 đạt 0,8533, tốt hơn khoảng 0,67 điểm phần trăm trên tập kiểm tra này. Đây là so sánh hai cách cắt tỉa đã chọn, không dùng điểm kiểm tra để dò từng giá trị tham số.

## Bài 5 – Độ quan trọng bằng xáo trộn



Cột ngẫu nhiên đứng thứ 3 theo Gini nhưng xuống cuối theo permutation. Gini importance đo lượng giảm độ lẫn lộn trên tập học; cột có nhiều giá trị có nhiều cơ hội chọn ngưỡng để tách cả nhiễu nên dễ nhận điểm cao. Permutation đo mức giảm accuracy trên tập kiểm tra khi xáo trộn một cột, vì vậy phản ánh khả năng đóng góp cho dự đoán ngoài tập học rõ hơn trong ví dụ này.

Điểm 0,0017 gần 0 và nhỏ hơn độ dao động giữa các lần xáo trộn: không nên kết luận cột ngẫu nhiên hữu ích chỉ vì điểm hơi dương. Độ quan trọng không chứng minh quan hệ nhân quả. Tập học dùng đúng chuỗi `default_rng(0)` của F6; tập kiểm tra lấy tiếp các số mới từ cùng bộ sinh, không dùng nhãn để sinh cột.

## Bài 6 – Rừng hồi quy



Rừng tốt hơn cây sâu 3 nhưng vẫn kém đường thẳng về MAE. Diện tích tập học từ 37,6 đến 117,5 m². Căn 250 m² và 350 m² đều vượt miền dữ liệu, cùng rơi vào lá ngoài cùng của mỗi cây. Mỗi lá đưa ra trung bình giá của các mẫu học trong lá, có tính số lần được lấy trong bootstrap; rừng lấy trung bình dự đoán của 200 cây. Vì vậy rừng không ngoại suy được xu hướng giá tăng theo diện tích như mô hình tuyến tính.
