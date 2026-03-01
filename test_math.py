from utils import calculate_angle

# 假設這是我們模擬的「標準橋式」數據
# 想像一個人平躺，身體呈一直線
# 肩膀(11) 在左邊，髖部(23) 在中間，膝蓋(25) 在右邊
shoulder = [100, 300]
hip = [200, 300]      # 屁股高度跟肩膀一樣 (平的)
knee = [300, 300]

# 呼叫我們剛剛寫好的公式
angle = calculate_angle(shoulder, hip, knee)

print("---------------------------")
print(f"肩膀座標: {shoulder}")
print(f"髖部座標: {hip}")
print(f"膝蓋座標: {knee}")
print(f"計算出的髖部角度: {int(angle)} 度")
print("---------------------------")

# 簡單的判斷邏輯
if angle > 170:
    print("判定：姿勢標準 (一直線)！")
else:
    print("判定：屁股掉下來了！")