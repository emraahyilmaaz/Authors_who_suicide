import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

driver = webdriver.Chrome()

driver.get('https://1000kitap.com/yazarlar/en-cok-okunanlar')

# YAZARLARIN LİNKLERİNİ AL
max_page = 140
author_urls = []
author_link_xpath = "//div[contains(@class, 'dr flex-1 gap-1')]/a"

for i in range(1, max_page):
    driver.get(f'https://1000kitap.com/yazarlar/en-cok-okunanlar?sayfa={i}')
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    author_element = driver.find_elements(By.XPATH, author_link_xpath)
    temp_url = [element.get_attribute('href') for element in author_element]
    author_urls.extend(temp_url)


data = []
# DATA ÇEKME İŞLEMİ
for i in range(len(author_urls)):
    try:
        driver.get(author_urls[i])
        time.sleep(5)
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

        time.sleep(5)

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
            'link': i
        }

        data.append(author_data)
        time.sleep(5)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

authors_details = pd.DataFrame(data)
authors_details.to_csv('authors_detail.csv', index=False)

len(authors_details)

authors_details['links'] = authors_details['link'].apply(lambda x: author_urls[x])

authors_df = authors_details
authors_df.to_csv('author_name.csv', index=False)

authors_df = pd.read_csv('author_name.csv')

link = authors_df['links'].tolist()
books_info = []

driver = webdriver.Chrome()
books_info = []
for url in link:  # İlk 3 yazarı test etmek için
    try:
        driver.get(url)
        time.sleep(5)  # Sayfanın yüklenmesini bekle
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
            time.sleep(5)  # Sayfanın yüklenmesini bekle

            # İlk düğmeye tıklama
            first_button_xpath = '//div[@class="dr svg w-5 h-text-20"]'
            first_button = driver.find_elements(By.XPATH, first_button_xpath)

            if first_button:
                script = "arguments[0].click();"
                driver.execute_script(script, first_button[1])  # 2. elemana tıkla
                time.sleep(5)  # Tıklamadan sonra bekle
            else:
                continue

            # İkinci düğmeye tıklama
            second_button_xpath = '//span[@class= "text font-medium text-14 text-silik"]'
            second_button = driver.find_elements(By.XPATH, second_button_xpath)

            if second_button:
                script = "arguments[0].click();"
                driver.execute_script(script, second_button[0])
                time.sleep(5)  # Tıklamadan sonra bekle
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

                time.sleep(2)  # Bilgileri aldıktan sonra bekle

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

df = pd.DataFrame(books_info)

df_pivot = pd.pivot_table(df, index='Name', columns='Book', values='About', aggfunc=lambda x: ' '.join(x))

df_pivot.to_csv('Author_books_detail.csv')


