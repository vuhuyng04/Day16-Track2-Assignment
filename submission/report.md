# Lab 16 — Báo cáo: LightGBM trên CPU (AWS)

1. Tôi dùng AWS, region `us-east-1`, Compute Node `t3.medium` (2 vCPU / 4 GB RAM, Ubuntu 22.04) trong private subnet, truy cập qua Bastion `t3.micro`; hạ tầng triển khai bằng Terraform, source commit `55539f6`.
2. Dataset Credit Card Fraud Detection (Kaggle `mlg-ulb/creditcardfraud`) có 284,807 dòng × 31 cột (0.17% là gian lận); chia stratified 80% train / 20% test, rồi tách 10% của train làm validation cho early stopping, seed `42`.
3. Load dữ liệu mất 2.50 giây; training mất 5.52 giây; best iteration là 64 (early stopping 100 vòng theo AUC trên validation, `learning_rate=0.05`, `num_leaves=31`).
4. AUC 0.9767, Accuracy 0.9995, F1 0.8287, Precision 0.9036, Recall 0.7653 trên tập test — Accuracy gần như 100% chủ yếu do dữ liệu mất cân bằng, nên F1/Precision/Recall và AUC mới là chỉ số có ý nghĩa.
5. Latency 1 dòng 1.25 ms; throughput batch 1,000 dòng ≈ 319,961 dòng/giây (0.0031 s/1,000 dòng); latency đo bằng trung bình 100 lần `predict_proba` 1 dòng sau 10 lần warm-up, throughput đo 1 lần `predict_proba` trên 1,000 dòng — latency 1 dòng bị chi phối bởi overhead Python/pandas, batch inference nhanh hơn hàng trăm lần mỗi dòng.
6. CPU/RAM/Network tôi quan sát sau khi chạy benchmark (02/10/2026 ~17:05, node ở trạng thái nghỉ): CPU ~99.8% idle (training chỉ ~5.5 s nên không bắt kịp lúc tải đỉnh), RAM used ~270 Mi / 3.7 Gi (available 3.2 Gi, buff/cache 1.7 Gi do đọc file CSV 144 MB), Network `ens5` RX ~277 MB / TX ~1.2 MB (chủ yếu tải package pip và dataset qua NAT); ảnh đính kèm `screenshots/02_top.png`, `03_free_h.png`, `04_network.png`.
7. Billing tại 02/10/2026 17:12 chưa cập nhật chi phí tháng 10 (AWS trễ 8–24h), ảnh `screenshots/05_billing.png`; ước tính ~$0.14/giờ (t3.medium $0.0416 + t3.micro $0.0104 + NAT $0.045 + ALB $0.0225 + 4 Public IPv4 $0.02 + EBS ~$0.002) + ~$0.02 data qua NAT; lab chạy ~1.25 giờ (16:00 → 17:15) ≈ **$0.20–0.25** (NAT/ALB tính theo giờ tròn).
8. Tôi đã tải kết quả về `submission/` và xóa tài nguyên bằng `terraform destroy` lúc 02/10/2026 17:15 (`Destroy complete! Resources: 27 destroyed.`); `terraform state list` trả về rỗng; kiểm tra AWS CLI tại us-east-1: 0 EC2 instance, 0 NAT Gateway, 0 ALB, 0 Elastic IP, 0 EBS volume.

**Nhận xét:** Với dữ liệu dạng bảng ~285K dòng, LightGBM trên 2 vCPU train khoảng 5.5 giây và đạt AUC ~0.977 — CPU nhỏ, rẻ là đủ, không cần GPU. Chi phí chủ yếu đến từ NAT Gateway chứ không phải compute, nên phải destroy ngay sau khi làm xong.

## Bảng kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.498 s |
| Thời gian training | 5.524 s |
| Best iteration | 64 |
| AUC-ROC | 0.976684 |
| Accuracy | 0.999456 |
| F1-Score | 0.828729 |
| Precision | 0.903614 |
| Recall | 0.765306 |
| Inference latency (1 row) | 1.251 ms |
| Inference throughput (1000 rows) | 0.0031 s (~319,961 rows/s) |
