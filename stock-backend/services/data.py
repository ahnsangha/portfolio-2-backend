# 종목 메타데이터 레이어
# 한국 상장사 기본 정보(코드, 이름, 시장)를 메모리에 캐싱해 빠르게 제공
# 외부 데이터 호출 위치를 한 곳으로 모아 사용처가 단순해지도록 구성

from datetime import datetime, timedelta # 마지막 갱신 시점과 주기 계산에 사용
from functools import lru_cache # 전체 목록을 간단 캐싱하기 위해 사용
from typing import Optional, List # 선택형 타입과 리스트 타입 표기에 사용
import pandas as pd, FinanceDataReader as fdr # 표 처리와 상장사 목록 수집에 사용

class KoreanStockDataManager:
    # 한국 상장사 메타데이터를 관리하는 클래스
    def __init__(self):
        self.stock_list: Optional[List[dict]] = None # 종목 목록 캐시 저장소
        self.last_update: Optional[datetime] = None # 마지막 갱신 시각 기록
        self.update_interval = timedelta(days=1) # 목록 갱신 주기 기본값

    @lru_cache(maxsize=1) # 동일 인자 호출 시 결과를 메모리에 보관해 재사용
    def get_all_stocks(self, force_update=False):
        # 캐시가 있고 갱신 주기 안이면 저장된 목록을 바로 반환
        if (not force_update and self.stock_list
            and self.last_update
            and datetime.now() - self.last_update < self.update_interval):
            return self.stock_list

        # 코스피와 코스닥 상장 목록을 읽어옴
        kospi = fdr.StockListing("KOSPI"); kospi["Market"] = "KOSPI" # 코스피 시장 정보 추가
        kosdaq = fdr.StockListing("KOSDAQ"); kosdaq["Market"] = "KOSDAQ" # 코스닥 시장 정보 추가
        # 두 표를 합치고 필요한 열만 남김
        all_stocks = pd.concat([kospi, kosdaq], ignore_index=True)[["Code", "Name", "Market"]]
        # 열 이름을 코드, 이름, 시장으로 통일
        all_stocks.rename(columns={"Code": "code", "Name": "name", "Market": "market"}, inplace=True)
        # 티커를 6자리 코드에 거래소 접미사 붙여 생성
        all_stocks["ticker"] = all_stocks["code"].str.zfill(6) + ".KS"
        # 딕셔너리 목록으로 변환해 캐시에 저장
        self.stock_list = all_stocks.to_dict("records")
        # 갱신 시각 기록
        self.last_update = datetime.now()
        # 준비된 종목 목록 반환
        return self.stock_list

    # 백업용 대표 종목 목록
    def _get_default_stocks(self):
        return [
            {'ticker': '005930.KS', 'code': '005930', 'name': '삼성전자', 'market': 'KOSPI'},
            {'ticker': '000660.KS', 'code': '000660', 'name': 'SK하이닉스', 'market': 'KOSPI'},
            {'ticker': '035420.KS', 'code': '035420', 'name': 'NAVER', 'market': 'KOSPI'},
            {'ticker': '035720.KS', 'code': '035720', 'name': '카카오', 'market': 'KOSPI'},
            {'ticker': '051910.KS', 'code': '051910', 'name': 'LG화학', 'market': 'KOSPI'},
            {'ticker': '005380.KS', 'code': '005380', 'name': '현대차', 'market': 'KOSPI'},
            {'ticker': '005490.KS', 'code': '005490', 'name': 'POSCO홀딩스', 'market': 'KOSPI'},
            {'ticker': '105560.KS', 'code': '105560', 'name': 'KB금융', 'market': 'KOSPI'},
        ]
    
    # 종목 검색 함수
    def search_stocks(self, query: str, market: Optional[str] = None, limit: int = 50):
        # 캐시가 비어 있으면 전체 목록을 먼저 채움
        if not self.stock_list:
            self.get_all_stocks()
        
        # 검색 편의를 위해 소문자로 통일
        query = query.lower()
        results = []
        
        # 캐시에 있는 모든 종목을 순회하며 조건에 맞는 항목만 담음
        for stock in self.stock_list:
            # 시장 필터가 있으면 해당 시장만 통과
            if market and market.lower() != 'all' and stock['market'].lower() != market.lower():
                continue
            
            # 이름 또는 코드에 검색어가 포함되면 결과에 추가
            if (query in stock['name'].lower() or 
                query in stock['code']):
                results.append(stock)
            
            # 결과 개수가 제한 수에 도달하면 중단
            if len(results) >= limit:
                break
        
        # 최종 검색 결과 반환
        return results

# 애플리케이션 전역에서 재사용할 단일 인스턴스
stock_manager = KoreanStockDataManager()
