import re
import unicodedata

from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)


def unbold_text(text):
    # Mapping of bold numbers to their regular equivalents
    bold_numbers = {
        "𝟬": "0",
        "𝟭": "1",
        "𝟮": "2",
        "𝟯": "3",
        "𝟰": "4",
        "𝟱": "5",
        "𝟲": "6",
        "𝟳": "7",
        "𝟴": "8",
        "𝟵": "9",
    }

    # Function to convert bold characters (letters and numbers)
    def convert_bold_char(match):
        char = match.group(0)
        # Convert bold numbers
        if char in bold_numbers:
            return bold_numbers[char]
        # Convert bold uppercase letters
        elif "\U0001d5d4" <= char <= "\U0001d5ed":
            return chr(ord(char) - 0x1D5D4 + ord("A"))
        # Convert bold lowercase letters
        elif "\U0001d5ee" <= char <= "\U0001d607":
            return chr(ord(char) - 0x1D5EE + ord("a"))
        else:
            return char  # Return the character unchanged if it's not a bold number or letter

    # Regex for bold characters (numbers, uppercase, and lowercase letters)
    bold_pattern = re.compile(
        r"[\U0001D5D4-\U0001D5ED\U0001D5EE-\U0001D607\U0001D7CE-\U0001D7FF]"
    )
    text = bold_pattern.sub(convert_bold_char, text)

    return text


def unitalic_text(text):
    # Function to convert italic characters (both letters)
    def convert_italic_char(match):
        char = match.group(0)
        # Unicode ranges for italic characters
        if "\U0001d608" <= char <= "\U0001d621":  # Italic uppercase A-Z
            return chr(ord(char) - 0x1D608 + ord("A"))
        elif "\U0001d622" <= char <= "\U0001d63b":  # Italic lowercase a-z
            return chr(ord(char) - 0x1D622 + ord("a"))
        else:
            return char  # Return the character unchanged if it's not an italic letter

    # Regex for italic characters (uppercase and lowercase letters)
    italic_pattern = re.compile(r"[\U0001D608-\U0001D621\U0001D622-\U0001D63B]")
    text = italic_pattern.sub(convert_italic_char, text)

    return text


def remove_emojis_and_symbols(text):
    # Extended pattern to include specific symbols like ↓ (U+2193) or ↳ (U+21B3)
    emoji_and_symbol_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002193"  # downwards arrow
        "\U000021b3"  # downwards arrow with tip rightwards
        "\U00002192"  # rightwards arrow
        "]+",
        flags=re.UNICODE,
    )

    return emoji_and_symbol_pattern.sub(r" ", text)


def replace_urls_with_placeholder(text, placeholder="[URL]"):
    # Regular expression pattern for matching URLs
    url_pattern = r"https?://\S+|www\.\S+"

    return re.sub(url_pattern, placeholder, text)


def remove_non_ascii(text: str) -> str:
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def clean_text(text_content: str | None) -> str:
    if text_content is None:
        return ""

    cleaned_text = unbold_text(text_content)
    cleaned_text = unitalic_text(cleaned_text)
    cleaned_text = remove_emojis_and_symbols(cleaned_text)
    cleaned_text = clean(cleaned_text)
    cleaned_text = replace_unicode_quotes(cleaned_text)
    
    # Chuẩn hóa Unicode sang NFC
    # Điều này đảm bảo các ký tự có dấu được biểu diễn nhất quán.
    cleaned_text = unicodedata.normalize("NFC", cleaned_text)
    
    # Hàm này loại bỏ dấu tiếng việt, chỉ sử dụng với tiếng Anh
    # cleaned_text = clean_non_ascii_chars(cleaned_text)
    
    # Trong mathpal, các URL chính là đường dẫn đến ảnh của các bài kiểm tra, đó là một thông tin quan trọng cần giữ
    # cleaned_text = replace_urls_with_placeholder(cleaned_text) 

    return cleaned_text


#=============== TESTING =================
def _write_output_to_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)

if __name__ == "__main__":
    raw_content = {
  "_id": "285e8c25-68c0-426f-b178-e43c7476fcb5",
  "content": "**Đáp án**\n**HƯỚNG DẪN GIẢI CHI TIẾT**\n**Phần 1: Trắc nghiệm**\n**Câu 1.** Tìm $x$ sao cho $\\frac{{12}}{5} < \\overline {x,2} < \\frac{{13}}{4}$\nA. 5 \nB. 2 \nC. 3 \nD. 4\n**Cách giải**\nTa có $\\frac{{12}}{5} < \\overline {x,2} < \\frac{{13}}{4}$ nên 2,4 < $\\overline {x,2} $ < 3,25\nDo đó $2 < x \\leqslant 3$. Suy ra $x = 3$\nChọn **C**\n**Câu 2.** An đi học lúc 6 giờ 45 phút, xe bus di chuyển hết 10 phút, thời gian chờ xe bus là 0,5 giờ. Hỏi An đến trường lúc mấy giờ?\nA. 7 giờ 25 phút \nB. 8 giờ 25 phút\nC. 7 giờ 15 phút \nD. 7 giờ 35 phút\n**Cách giải**\nĐổi 0,5 giờ = 30 phút\nAn đến trường lúc: 6 giờ 45 phút + 10 phút + 30 phút = 7 giờ 25 phút\nChọn **A**\n**Câu 3.** Tổ gồm 10 công nhân hoàn thành xong một công việc trong 30 ngày. Nếu tổ có 20 công nhân thì hoàn thành công việc trong thời gian bao lâu?\nA. 10 ngày \nB. 60 ngày \nC. 40 ngày \nD. 15 ngày\n**Cách giải**\n1 công nhân hoàn thành công việc trong thời gian là 30 x 10 = 300 (ngày)\n20 công nhân hoàn thành công việc trong thời gian là 300 : 20 = 15 (ngày)\nChọn **D**\n**Câu 4.** Phòng học có dạng hình hộp chữ nhật chiều dài 7m, chiều rộng 4,5m, chiều cao 3m. Người ta muốn sơn toàn bộ trần nhà và 4 bức tường. Biết tổng diện tích các cửa là 7,5m2, tính diện tích cần sơn.\nA. 83m2 \nB. 108 m2 \nC. 93 m2\nD. 98 m2\n**Cách giải**\nDiện tích xung quanh của phòng học là (7 + 4,5) x 2 x 3 = 69 (m2)\nDiện tích trần nhà là: 7 x 4,5 = 31,5 (m2)\nDiện tích cần sơn là 69 + 31,5 – 7,5 = 93 (m2)\nChọn **C**\n**Phần 2: Điền đáp số**\n**Câu 5.** Cho A = $\\overline {52xy} $. Biết A chia hết cho 2 và 9; chia 5 dư 4. Tìm A.\n**Cách giải**\nVì A chia hết cho 2 và chia 5 dư 4 nên y = 4\nTa có số A = $\\overline {52x4} $\nVì A chia hết cho 9 nên 5 + 2 + x + 4 chia hết cho 9. Suy ra x = 7\nVậy A = 5274\n**Câu 6.** Tổng số gạo kho I và kho II là 46 tấn. Biết 15 lần số gạo kho I bằng 8 lần số gạo kho II. Hỏi kho II chứa bao nhiêu tấn gạo?\n**Cách giải**\nVì 15 lần số gạo kho I\nKho II chứa số tấn gạo là:\n46 : (8 + 15) x 15 = 30 (tấn gạo)\nĐáp số: 30 tấn gạo\n**Câu 7.** Tính tỉ số phần trăm số học sinh thích ăn cam và chuối so với số học sinh thích ăn táo và xoài?\n![](https://img.loigiaihay.com/picture/2023/0728/20_15.png)\n**Cách giải**\nSố học sinh thích ăn cam và chuối là: 4 + 3 = 7 (học sinh)\nSố học sinh thích ăn táo và xoài là: 15 + 10 = 25 (học sinh)\nTỉ số phần trăm số học sinh thích ăn cam và chuối so với số học sinh thích ăn táo và xoài là:\n7 : 25 = 0,28 = 28%\nĐáp số: 28%\n**Câu 8.** Cho hình vuông ABCD. Vẽ nửa đường tròn đường kính AB và $\\frac{1}{4}$ đường tròn bán kính AB. Tính diện tích phần tô đậm biết chu vi đường tròn đường kính AB là 37,68 cm.\n![](https://img.loigiaihay.com/picture/2023/0728/20_17.png)\n**Cách giải**\nĐộ dài đoạn thẳng AB là 37,68 : 3,14 = 12 (cm)\nBán kính đường tròn đường kính AB là 12 : 2 = 6 (cm)\nDiện tích nửa đường tròn đường kính AB là 6 x 6 x 3,14 : 2 = 56,52 (cm2)\nDiện tích $\\frac{1}{4}$ đường tròn bán kính AB là 12 x 12 x 3,14 : 4 = 113,04 (cm2)\nDiện tích phần tô đậm là 113,04 – 56,52 = 56,52 (cm2)\nĐáp số: 56,52 cm2\n**Phần 3: Tự luận**\n**Bài 1.** Cho đoạn đường AD có 1 đoạn lên dốc, 1 đoạn xuống dốc, 1 đoạn bằng phẳng. Trong đó, đoạn đường AB = BC, CD = 4km (AB là đoạn đường lên dốc, BC là đoạn đường xuống dốc, CD là đoạn đường bằng phẳng). Biết vận tốc khi lên dốc là 4 km/giờ, vận tốc khi xuống dốc là là 6 km/giờ, vận tốc đi trên đoạn đường bằng phẳng là 5 km/giờ.\na) Tính thời gian đi đoạn đường CD.\nb) Tính độ dài đoạn đường AD, biết người đó đi từ A lúc 5 giờ và đến D lúc 6 giờ 48 phút.\n**Cách giải**\na) Thời gian đi đoạn đường CD là $4:5 = \\frac{4}{5}$ (giờ) = 48 (phút)\nb) Thời gian đi hết đoạn đường AC là 6 giờ 48 – 5 – 48 phút = 1 (giờ)\nTỉ số vận tốc trên đoạn đường AB so với đoạn đường BC $4:6 = \\frac{2}{3}$\nTrên cùng một quãng đường thì vận tốc tỉ lệ nghịch với thời gian.\nVì đoạn đường AB = BC nên tỉ số thời gian đi hết đoạn đường AB so với đoạn đường BC là $\\frac{3}{2}$\nThời gian đi hết đoạn đường AB là 1 : (2 + 3) x 3 = 0,6 (giờ)\nĐộ dài đoạn đường AB là 0,6 x 4 = 2,4 (km)\nĐộ dài đoạn đường AD là 2,4 x 2 + 4 = 8,8 (km)\nĐáp số: a) 48 phút\nb) 8,8 km\n**Bài 2**. Bạn Hưng viết 5 số tự nhiên khác nhau trên một vòng tròn sao cho không có 2 hoặc 3 số nào ở vị trí liên tiếp nhau có tổng chia hết cho 3.\n![](https://img.loigiaihay.com/picture/2023/0728/19_3.png)\na) Hãy tìm một bộ 5 số tự nhiên thỏa mãn yêu cầu đề bài và điền vào hình.\nb) Chứng tỏ trong mỗi bộ 5 số thỏa mãn yêu cầu đề bài có ít nhất 1 số chia hết cho 3.\n**Cách giải:**\na) Bộ 5 số thỏa mãn yêu cầu như hình vẽ:\n![](https://img.loigiaihay.com/picture/2023/0620/19_1.png)\nb) Giả sử cả 5 số đều không chia hết cho 3, khi đó các số này chia 3 dư 1 hoặc chia 3 dư 2.\n- Xét số a chia 3 dư 1. Khi đó để tổng hai số liên tiếp (có chứa số a) không chia hết cho 3 thì hai số đứng cạnh a phải chia 3 dư 1. Do đó, ta có 3 số liên tiếp cùng chia 3 dư 1, tức là tổng 3 số này chia hết cho 3 (vô lí)\n- Lập luận tương tự với số chia 3 dư 2, ta thấy vô lí.\nVậy trong 5 số luôn có ít nhất 1 số phải chia hết cho 3.\n",
  "link": "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html",
  "grade_id": "2cc4ad3a-9d7f-42e4-bf2e-409b8d68c7ea"
}
    print(raw_content["content"])
    _write_output_to_file("raw_text.txt", raw_content["content"])
    print("="*60)
    cleaned_content = clean_text(raw_content["content"])
    print(cleaned_content)
    _write_output_to_file("cleaned_text.txt", cleaned_content)