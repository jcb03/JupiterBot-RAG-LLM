import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
import hashlib
from urllib.parse import urljoin, urlparse
from datetime import datetime
import streamlit as st
from .config import Config

class JupiterWebScraper:
    """Enhanced web scraper for Jupiter's website"""
    
    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.scraped_urls = set()
        self.scraped_data = []
        
    def is_valid_jupiter_url(self, url):
        """Check if URL belongs to Jupiter domain"""
        try:
            parsed = urlparse(url)
            domain_valid = parsed.netloc.lower() in ['jupiter.money', 'www.jupiter.money']
            
            skip_patterns = [
                'javascript:', 'mailto:', 'tel:', '#',
                '.pdf', '.jpg', '.png', '.gif', '.svg',
                'download', 'api/', 'admin', 'login', 'register'
            ]
            
            url_clean = url.lower()
            should_skip = any(pattern in url_clean for pattern in skip_patterns)
            
            return domain_valid and not should_skip and len(url) < 200
            
        except Exception:
            return False
    
    def clean_text_content(self, text):
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted phrases
        unwanted_phrases = [
            'This site uses cookies', 'Accept cookies', 'Cookie Policy',
            'Privacy Policy', '© Jupiter', 'All rights reserved'
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, '')
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Clean punctuation
        text = re.sub(r'[^\w\s.,!?()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, title, content):
        """Extract relevant keywords"""
        full_text = f"{title} {content}".lower()
        
        important_keywords = {
            'account', 'banking', 'savings', 'investment', 'credit', 'debit',
            'loan', 'insurance', 'money', 'payment', 'transfer', 'jupiter',
            'app', 'digital', 'fintech', 'fees', 'charges', 'kyc',
            'verification', 'security', 'upi', 'neft', 'rtgs', 'fund'
        }
        
        found_keywords = []
        for keyword in important_keywords:
            if keyword in full_text:
                found_keywords.append(keyword)
        
        return found_keywords[:10]
    
    def determine_content_category(self, url, title, content):
        """Categorize content"""
        url_lower = url.lower()
        
        url_categories = {
            'faq': ['/faq', '/help', '/support'],
            'legal': ['/privacy', '/terms', '/legal'],
            'product': ['/features', '/services', '/products'],
            'about': ['/about', '/company', '/team'],
            'pricing': ['/pricing', '/fees', '/charges'],
            'security': ['/security', '/safety'],
            'blog': ['/blog', '/news', '/article']
        }
        
        for category, patterns in url_categories.items():
            if any(pattern in url_lower for pattern in patterns):
                return category
        
        return 'general'
    
    def scrape_single_page(self, url):
        """Scrape content from a single webpage"""
        try:
            print(f"🔍 Scraping: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                               'iframe', 'noscript']):
                element.decompose()
            
            # Extract title
            title = ""
            title_element = soup.find('title')
            if title_element:
                title = title_element.get_text().strip()
            
            if not title or len(title) < 10:
                h1_element = soup.find('h1')
                if h1_element:
                    title = h1_element.get_text().strip()
            
            # Extract main content
            content = ""
            main_selectors = [
                'main', 'article', '[role="main"]', '.main-content',
                '.content', '.post-content', '.entry-content'
            ]
            
            for selector in main_selectors:
                elements = soup.select(selector)
                if elements:
                    content_parts = []
                    for elem in elements:
                        text = elem.get_text(separator=' ', strip=True)
                        if text and len(text) > 50:
                            content_parts.append(text)
                    content = ' '.join(content_parts)
                    break
            
            if not content.strip():
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            content = self.clean_text_content(content)
            
            if len(content) < 200:
                print(f"⚠️ Skipping {url} - insufficient content")
                return None
            
            # Generate metadata
            category = self.determine_content_category(url, title, content)
            keywords = self.extract_keywords(title, content)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            scraped_data = {
                'url': url,
                'title': title or "Untitled Page",
                'content': content,
                'content_hash': content_hash,
                'category': category,
                'keywords': keywords,
                'length': len(content),
                'word_count': len(content.split()),
                'scraped_at': datetime.now().isoformat()
            }
            
            print(f"✅ Successfully scraped: {title[:50]}...")
            return scraped_data
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    def find_internal_links(self, url, soup, max_links=8):
        """Find relevant internal links"""
        links = set()
        
        priority_patterns = [
            'features', 'about', 'help', 'faq', 'pricing', 'services',
            'security', 'support', 'products', 'banking'
        ]
        
        for link in soup.find_all('a', href=True):
            if len(links) >= max_links:
                break
                
            href = link['href']
            full_url = urljoin(url, href)
            
            if (self.is_valid_jupiter_url(full_url) and 
                full_url not in self.scraped_urls):
                
                link_text = link.get_text().lower()
                is_priority = any(pattern in link_text or pattern in full_url.lower() 
                                for pattern in priority_patterns)
                
                if is_priority or len(links) < max_links // 2:
                    links.add(full_url)
        
        return list(links)
    
    def scrape_jupiter_website(self, progress_callback=None):
        """Main scraping function"""
        print("🚀 Starting Jupiter website scraping...")
        
        start_urls = [
                f"{self.config.JUPITER_BASE_URL}",
                f"{self.config.JUPITER_BASE_URL}/about-us",           
                f"{self.config.JUPITER_BASE_URL}/fees-rates-charges", 
                f"{self.config.JUPITER_BASE_URL}/contact-us",         
                f"{self.config.JUPITER_BASE_URL}/savings-account",   
                f"{self.config.JUPITER_BASE_URL}/pro-salary-account", 
                f"{self.config.JUPITER_BASE_URL}/edge-csb-rupay-credit-card",
                f"{self.config.JUPITER_BASE_URL}/edge-plus-upi-rupay-credit-card",
                f"{self.config.JUPITER_BASE_URL}/edge-visa-credit-card"
        ]
        
        urls_to_scrape = list(start_urls)
        pages_scraped = 0
        successful_scrapes = 0
        max_pages = self.config.MAX_PAGES_TO_SCRAPE
        
        while urls_to_scrape and pages_scraped < max_pages:
            url = urls_to_scrape.pop(0)
            
            if url in self.scraped_urls:
                continue
            
            self.scraped_urls.add(url)
            pages_scraped += 1
            
            if progress_callback:
                progress = min(pages_scraped / max_pages, 1.0)
                progress_callback(progress, f"Scraping page {pages_scraped}/{max_pages}")
            
            page_data = self.scrape_single_page(url)
            
            if page_data:
                self.scraped_data.append(page_data)
                successful_scrapes += 1
                
                if successful_scrapes <= 6:
                    try:
                        response = self.session.get(url, timeout=10)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        new_links = self.find_internal_links(url, soup)
                        urls_to_scrape.extend(new_links)
                    except:
                        pass
            
            time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
        
        # Save data
        os.makedirs('data', exist_ok=True)
        output_file = 'data/scraped_content.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Scraping completed! {successful_scrapes} pages scraped")
        
        return {
            'success': True,
            'pages_scraped': successful_scrapes,
            'total_attempted': pages_scraped,
            'data_file': output_file,
            'scraped_data': self.scraped_data
        }

def run_scraping_with_progress():
    """Run scraping with Streamlit progress bar"""
    scraper = JupiterWebScraper()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    
    try:
        result = scraper.scrape_jupiter_website(progress_callback=update_progress)
        progress_bar.progress(1.0)
        status_text.text("✅ Scraping completed!")
        return result
    except Exception as e:
        status_text.text(f"❌ Scraping failed: {str(e)}")
        return {'success': False, 'error': str(e)}
