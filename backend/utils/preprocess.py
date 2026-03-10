import os
import cv2
import numpy as np
import torch


def preprocess_image(image_bytes, save_debug=True, debug_dir="debug_images"):
    # debug 폴더 생성
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    # 1. 바이트 데이터를 numpy 배열로 변환
    file_bytes = np.frombuffer(image_bytes, np.uint8)

    # 2. 이미지 디코딩 (컬러 이미지로 읽기)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "1_original.png"), image)

    # 3. grayscale 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "2_gray.png"), gray)

    # 4. 블러로 노이즈 약간 줄이기
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "3_blurred.png"), blurred)

    # 5. threshold
    # 흰 종이 위 검은 숫자라고 가정
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "4_threshold.png"), thresh)

    # 6. threshold 후 작은 노이즈 정리
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "4_1_cleaned.png"), cleaned)

    # 7. connected components 분석
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    h_img, w_img = cleaned.shape
    best_idx = -1
    best_area = 0

    for i in range(1, num_labels):  # 0은 background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 너무 작은 건 제외
        if area < 30:
            continue

        # 너무 큰 건 제외
        if area > h_img * w_img * 0.2:
            continue

        # 테두리에 붙은 건 제외 (배경/종이/그림자 잡는 경우 많음)
        if x <= 1 or y <= 1 or (x + w) >= w_img - 1 or (y + h) >= h_img - 1:
            continue

        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx == -1:
        raise ValueError("유효한 숫자 영역을 찾을 수 없습니다.")

    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]

    # 너무 작은 영역이면 오류 처리
    if w < 10 or h < 10:
        raise ValueError("숫자 영역이 너무 작습니다.")

    # bounding box 시각화
    if save_debug:
        boxed = image.copy()
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "5_bounding_box.png"), boxed)

    # 8. 숫자 영역 crop
    digit = cleaned[y:y+h, x:x+w]
    digit = cv2.dilate(digit, np.ones((3, 3), np.uint8), iterations=2)
    

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "6_digit_crop.png"), digit)

    # 9. 정사각형 캔버스 만들기
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)

    # 숫자를 가운데 배치
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "7_square.png"), square)

    # 10. 가장자리 여백 추가
    padded = cv2.copyMakeBorder(
        square,
        2, 2, 2, 2,
        cv2.BORDER_CONSTANT,
        value=0
    )

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "8_padded.png"), padded)

    # 11. 28x28 resize
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "9_resized_28x28.png"), resized)

    # 12. 0~1 범위로 정규화
    resized = resized.astype(np.float32) / 255.0

    # 디버그용: 정규화 전 사람이 보기 쉽게 다시 저장
    if save_debug:
        viewable = (resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "10_viewable_before_mnist_norm.png"), viewable)

    # 13. MNIST 정규화
    resized = (resized - 0.1307) / 0.3081

    # 14. tensor shape 맞추기: (1, 1, 28, 28)
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor