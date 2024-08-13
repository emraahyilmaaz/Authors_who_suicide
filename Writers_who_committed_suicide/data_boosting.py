import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

time_out = 3

new_author_links = ['https://1000kitap.com/yazar/Walter-Benjamin?hl=tr',
        'https://1000kitap.com/yazar/heinrich-von-kleist?hl=tr', 'https://1000kitap.com/yazar/mariano-jose-de-larra?hl=tr',
        'https://1000kitap.com/yazar/kaan-ince?hl=tr', 'https://1000kitap.com/yazar/metin-kacan?hl=tr',
        'https://1000kitap.com/yazar/Kanat-Guner?hl=tr', 'https://1000kitap.com/yazar/arthur-koestler?hl=tr',
                    'https://1000kitap.com/yazar/kurt-tucholsky?hl=tr', 'https://1000kitap.com/yazar/ned-vizzini?hl=tr',
                    'https://1000kitap.com/yazar/vladimir-mayakovski?hl=tr', 'https://1000kitap.com/yazar/klaus-mann?hl=tr',
                 'https://1000kitap.com/yazar/Guy-debord?hl=tr', 'https://1000kitap.com/yazar/a-adamov?hl=tr',
        'https://1000kitap.com/yazar/john-kennedy-toole', 'https://1000kitap.com/yazar/yasunari-kawabata?hl=tr',
    'https://1000kitap.com/yazar/david-foster-wallace', 'https://1000kitap.com/yazar/yukio-misima?hl=tr',
                    'https://1000kitap.com/yazar/richard-brautigan?hl=tr', 'https://1000kitap.com/yazar/primo-levi?hl=tr',
                    'https://1000kitap.com/yazar/besir-fuad', 'https://1000kitap.com/yazar/Evan-Wright?hl=tr',
                    'https://1000kitap.com/yazar/Robert-E-Howard?hl=tr', 'https://1000kitap.com/yazar/hunter-s-thompson?hl=tr',
                   'https://1000kitap.com/yazar/tadeusz-borowski']

driver = webdriver.Chrome()

data = []

# DATA ÇEKME İŞLEMİ
for url in new_author_links:
    try:
        driver.get(url)
        time.sleep(time_out)
        name = driver.find_elements(By.XPATH, '//h1')[1]

        rating_xpath = '//span[@class="text font-medium text-15"]'
        rating_elements = driver.find_elements(By.XPATH, rating_xpath)
        rating = rating_elements[0].text if rating_elements else None

        read_xpath = "//span[@class='text text-15']"
        read_elements = driver.find_elements(By.XPATH, read_xpath)
        read = read_elements[0].text if len(read_elements) > 0 else None
        liked = read_elements[1].text if len(read_elements) > 1 else None
        seen = read_elements[2].text if len(read_elements) > 2 else None

        # ölüm, unvan, doğum ve isim:
        born_xpath = "//div[@class='dr w-18']/span[contains(text(), 'Doğum')]"
        death_xpath = "//div[@class='dr w-18']/span[contains(text(), 'Ölüm')]"

        born_elements = driver.find_elements(By.XPATH, born_xpath)
        born = born_elements[0].find_element(By.XPATH, "../following-sibling::div[@class='dr flex-1']/span").text \
            if born_elements else None

        death_elements = driver.find_elements(By.XPATH, death_xpath)
        death = death_elements[0].find_element(By.XPATH, "../following-sibling::div[@class='dr flex-1']/span").text \
            if death_elements else None

        more_button_xpath = "//span[@class='text font-medium text-15 text-silik']"
        more_button = driver.find_elements(By.XPATH, more_button_xpath)

        if more_button:
            driver.execute_script("arguments[0].click();", more_button[0])
            time.sleep(5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        about_author_xpath = "//span[@class='text-alt']"
        about_author_elements = driver.find_elements(By.XPATH, about_author_xpath)
        about_author = about_author_elements[0].text if about_author_elements else "Hakkında bilgisi bulunamadı"

        time.sleep(time_out)

        # Yorumları almak için:
        comments_xpath = "//span[@class='text text text-16']"
        comments_elements = driver.find_elements(By.XPATH, comments_xpath)
        all_comments = [element.text for element in comments_elements]

        author_data = {
            'name': name.text,
            'rate': rating,
            'read': read,
            'liked': liked,
            'seen': seen,
            'born': born,
            'death': death,
            'about_author': about_author,
            'comment': all_comments,
            'link': url
        }

        data.append(author_data)
        time.sleep(5)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

authors_details = pd.DataFrame(data)
authors_details.to_csv('Boosting_authors_detail.csv', index=False)

time_out = 3
books_info = []

for url in new_author_links:  # İlk 3 yazarı test etmek için
    try:
        driver.get(url)
        time.sleep(time_out)  # Sayfanın yüklenmesini bekle
        name = driver.find_elements(By.XPATH, '//h1')[1]
        name_author = name.text

        # Kitap xpath
        books_xpath = '//a[@role="link" and contains(@class, "dr  mr-4 w-30 cursor")]'
        books = driver.find_elements(By.XPATH, books_xpath)

        # Kitap linkleri
        book_links = [element.get_attribute('href') for element in books]

        three_and_under_link = book_links[:3] if len(book_links) > 3 else book_links

        for index, book_link in enumerate(three_and_under_link):
            driver.get(book_link)
            time.sleep(time_out)  # Sayfanın yüklenmesini bekle

            # İlk düğmeye tıklama
            first_button_xpath = '//div[@class="dr svg w-5 h-text-20"]'
            first_button = driver.find_elements(By.XPATH, first_button_xpath)

            if first_button:
                script = "arguments[0].click();"
                driver.execute_script(script, first_button[1])  # 2. elemana tıkla
                time.sleep(time_out)  # Tıklamadan sonra bekle
            else:
                continue

            # İkinci düğmeye tıklama
            second_button_xpath = '//span[@class= "text font-medium text-14 text-silik"]'
            second_button = driver.find_elements(By.XPATH, second_button_xpath)

            if second_button:
                script = "arguments[0].click();"
                driver.execute_script(script, second_button[0])
                time.sleep(time_out)  # Tıklamadan sonra bekle
            else:
                continue

            # Kitap hakkında metni alma
            about_book_xpath = '//span[@class="text-alt"]'
            about_books = driver.find_elements(By.XPATH, about_book_xpath)

            if about_books:
                about_text = about_books[0].text
                book_info = {
                    'Name': name_author,
                    'Book': f'Book_{index + 1}',
                    'URL': book_link,
                    'About': about_text
                }
                books_info.append(book_info)

                time.sleep(time_out)  # Bilgileri aldıktan sonra bekle

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

df = pd.DataFrame(books_info)

df_pivot = pd.pivot_table(df, index='Name', columns='Book', values='About', aggfunc=lambda x: ' '.join(x))

df_pivot.to_csv('Boosting_author_books_detail.csv')


