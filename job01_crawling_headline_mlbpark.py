from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import re

# 각 팀의 URL 목록과 카테고리 매핑
urls = [
    ("KIA", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=KIA"),
    ("SAMSUNG", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=%EC%82%BC%EC%84%B1"),
    ("LG", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=LG"),
    ("DOOSAN", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=%EB%91%90%EC%82%B0"),
    ("KT", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=kt"),
    ("SSG", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=SSG"),
    ("LOTTE", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=%EB%A1%AF%EB%8D%B0"),
    ("HANWHA", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=%ED%95%9C%ED%99%94"),
    ("NC", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=NC"),
    ("KIWOOM", "https://mlbpark.donga.com/mp/b.php?search_select=sct&search_input=&select=spf&m=search&b=kbotown&query=%ED%82%A4%EC%9B%80")
]

def crawl_mlbpark_posts():
    # Chrome 옵션 설정
    options = ChromeOptions()
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # 웹드라이버 설정
    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    all_titles = []

    try:
        for category, url in urls:
            print(f"크롤링 시작: {category} - {url}")
            driver.get(url)

            # 페이지 로딩 대기
            time.sleep(5)

            # 게시글 제목 수집
            titles = []

            # 1페이지부터 10페이지까지 크롤링
            for page_num in range(1, 11):
                print(f"{page_num} 페이지 크롤링 시작")

                # 페이지 내 제목 요소 찾기
                title_elements = driver.find_elements(By.XPATH, '//a[contains(@class, "txt")]')

                for element in title_elements:
                    try:
                        title = element.text.strip()
                        title = re.compile('[^가-힣 ]').sub(' ', title)  # 한글과 띄어쓰기만 남기
                        titles.append((title, category))  # 제목과 카테고리 함께 저장
                    except Exception as e:
                        print(f"제목 추출 중 오류: {e}")

                # 다음 페이지로 이동 (1페이지는 기본 URL이고, 2페이지부터는 XPath로 이동)
                if page_num < 10:  # 마지막 페이지는 이동하지 않음
                    try:
                        next_page_xpath = f'//*[@id="container"]/div[3]/div[1]/div[2]/div[1]/a[{page_num + 1}]'
                        next_page_button = driver.find_element(By.XPATH, next_page_xpath)
                        next_page_button.click()  # 페이지 클릭하여 이동
                        time.sleep(5)  # 페이지 로딩 대기
                    except Exception as e:
                        print(f"페이지 이동 중 오류 발생: {e}")

                print(f"{page_num} 페이지에서 {len(title_elements)}개의 제목을 추출했습니다.")

            # 모든 제목들을 하나의 리스트에 추가
            all_titles.extend(titles)
            print(f"{category}에서 {len(titles)}개의 제목을 추출했습니다.")

        # 결과 출력 및 CSV 저장
        print(f"총 {len(all_titles)}개의 게시글 제목 수집 완료")

        # 데이터프레임 생성 및 CSV 저장
        df = pd.DataFrame(all_titles, columns=['titles', 'category'])
        df.to_csv('./crawling_data/mlbpark_titles_with_category.csv', index=False, encoding='utf-8-sig')

        # 제목 출력
        for title, category in all_titles:
            print(f"{category}: {title}")

        return df

    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")

    finally:zx
        #브라우저 종료
        driver.quit()

# 크롤링 실행
crawl_mlbpark_posts()
