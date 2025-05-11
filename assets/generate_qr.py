import qrcode
from PIL import Image

def generate_qr_code(url, filename="qr_code.png"):
    # QR 코드 생성
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # URL 추가
    qr.add_data(url)
    qr.make(fit=True)
    
    # QR 코드 이미지 생성
    qr_image = qr.make_image(fill_color="black", back_color="white")
    
    # 이미지 저장
    qr_image.save(filename)
    print(f"QR 코드가 {filename}로 저장되었습니다.")

if __name__ == "__main__":
    # GitHub 저장소 URL
    repo_url = "https://github.com/his0si/cnn-quantization-tflite"
    generate_qr_code(repo_url) 