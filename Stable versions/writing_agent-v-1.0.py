#This is an AI writing agent.
#It should perform the following actions.
#
# Database Structure:
# The SQLITE3 database should have the following table:
# CREATE TABLE content (
# id INTEGER PRIMARY KEY AUTOINCREMENT,
# title TEXT,
# url TEXT,
# file_path TEXT,
# content TEXT,
# source TEXT,
# date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# vector BLOB
# )

# Program Workflow:
# 0. Initialization:
# - Check if the --content flag is provided at runtime
# - If yes, create a temporary vectorized database:
# - Load all files from the specified folder (HTML, TXT, DOCX)
# - Process and vectorize the content
# - Store in a temporary SQLITE3 database with the same structure as the main database
# - Check if the --file flag is provided at runtime
# - If yes:
# - Load the specified file
# - Format the content for display (add [n] before each paragraph)
# - Go to the main menu
# - If no --file flag:
# - Go to the main menu directly

# Main Menu:
# - Display the current text (if any) formatted as follows:
# [1] (Paragraph)
# [2] (Paragraph)
# ...
# [n] (Paragraph)
# - Note: The brackets are shown only to make it easier for the user to select paragraphs,
# and should not be added to the actual text, unless they were actually there before.
# - Present options:
# 1. Research and write
# 2. Partial rewrite
# 3. Delete paragraph(s)
# 4. Save content into database
# 5. Exit

# 1. Writing and research.
# -Ask the user what they want to write about.
# -Ask for the number of keywords.
# -Detect keywords from text AND user requests.
# -Ask user: "Should we proceed? ([y]es [n]o [m]anual mode)".
# - If user selects yes, ask how many results should the AI should add to the context.
# - If the user selects no, go back to asking the number of keywords.
# - If invalid key, repeat question.
# - If user selects manual mode, ask for 1 keyword per line.
# -Browse results with Python's playwright library, and save them to the SQLITE3 content database:
# - Extract title, URL, and content from each result
# - Vectorize the content
# - Insert into the database (id, title, url, NULL, content, 'web', date_added, vector)
# -Use the vectorized database to find the most relevant information for the context.
# -Ask the AI to create a plan based on the vectorized search results and user's request.
# -Confirm with the user if the plan is correct ([y]es / [n]o / [c]ancel]. If not, ask what should be modified.
# -Repeat question until user selects yes, no or cancel.
# -If user selects 'yes', go to drafting stage. The AI should be aware of the text plan, summary and context.
# -If user selects 'no', ask what should be modified in the plan.
# -If the user selects 'cancel', go to main menu.

# 2. Partial rewrite
# - Display the current text with numbered paragraphs.
# - Prompt the user: "Enter the paragraph numbers you want to rewrite (e.g., 2 4-7 9):"
# - Parse the user input:
# - Split the input by spaces.
# - For each element:
# - If it's a single number, add it to the list of paragraphs to rewrite.
# - If it contains a dash, treat it as a range and add all numbers in that range.
# - For each selected paragraph or range:
# - Create a new AI instance for rewriting.
# - Detect keywords from text AND user requests.
# - Ask user: "Should we proceed? ([y]es [n]o [m]anual mode)".
# - If user selects yes, ask how many results should the AI should add to the context.
# - If the user selects no, go back to asking the number of keywords.
# - If invalid key, repeat question.
# - If user selects manual mode, ask for 1 keyword per line.
# - Use the vectorized SQLITE3 database to find the most relevant information for rewriting.
# - Ask the AI to rewrite the paragraph(s) while maintaining the original meaning and style
# - Show the original paragraph(s) in the "original paragraph" section (for the AI only, not the user)
# - Create a fact-checking instance:
# - Extract quotes from the original paragraph(s)
# - Compare each quote with the rewritten text
# - If a quote doesn't match exactly, discard it from the rewritten version
# - Display the rewritten paragraph(s) to the user
# - Prompt the user: "Accept this rewrite? ([y]es / [n]o / [e]dit)"
# - If 'yes', replace the original paragraph(s) with the rewritten version in the original text.
# DO NOT do with AI, but programmatically.
# - If 'no', keep the original paragraph(s) and move to the next selection.
# - After processing all selections, display the updated text.
# - Ask the user: "Do you want to make any more rewrites? ([y]es / [n]o)"
# - If 'yes', repeat the process.
# - If 'no', return to the main menu.

# 3. Delete paragraph(s)
# - Display the current text with numbered paragraphs.
# - Prompt the user: "Enter the paragraph numbers you want to delete (e.g., 2 4-7 9):"
# - Parse the user input:
# - Split the input by spaces.
# - For each element:
# - If it's a single number, add it to the list of paragraphs to delete.
# - If it contains a dash, treat it as a range and add all numbers in that range.
# - Confirm with the user: "Are you sure you want to delete these paragraphs? ([y]es / [n]o)"
# - If 'yes', delete the selected paragraphs from the text.
# - If 'no', return to the main menu.
# - Display the updated text.
# - Ask the user: "Do you want to delete more paragraphs? ([y]es / [n]o)"
# - If 'yes', repeat the process.
# - If 'no', return to the main menu.

# 4. Save content into database
# - Prompt the user for the file path or URL of the content to be saved
# - Detect if it's a local file or a URL using is_url() function
# - Process the content:
# - For local files:
# - Detect the file type (HTML, TXT, DOCX)
# - Read and extract text using process_file() function
# - Set file_path to the local path
# - Set url to NULL
# - Set source to 'local_file'
# - For URLs:
# - Use playwright to fetch the content using fetch_url_content() function
# - Set url to the provided URL
# - Set file_path to NULL
# - Set source to 'web'
# - Extract or generate a title for the content
# - Vectorize the content using vectorize_text() function
# - Insert the processed data into the SQLITE3 database:
# - For local files: (id, title, NULL, file_path, content, 'local_file', date_added, vector)
# - For URLs: (id, title, url, NULL, content, 'web', date_added, vector)
# - Confirm to the user that the content has been saved
# - Ask if the user wants to save more content or return to the main menu

# 5. Exit
# - Save any unsaved changes
# - Close the database connection
# - Terminate the program

# Helper Functions:
# - vectorize_text(text): Convert text to a vector representation
# - search_database(query, db_connection): Perform a similarity search on the vectorized database
# - process_file(file_path): Extract text from HTML, TXT, or DOCX files
# - fetch_url_content(url): Use playwright to fetch content from a URL
# - create_temp_database(): Create a temporary database from files in the specified folder
# - is_url(string): Check if the given string is a URL or a file path
# - format_text_for_display(text): Add [n] before each paragraph for display purposes
# - load_file(file_path): Load and process a file specified by the --file flag
import sqlite3
import os
import sys
import argparse
from datetime import datetime
from playwright.sync_api import sync_playwright
import docx
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import tempfile
import anthropic
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import validators

# Global vectorizer
vectorizer = TfidfVectorizer()

# Claude API setup
CLAUDE_API_KEY = ""
if not CLAUDE_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Database setup
def get_or_create_database(db_path):
    db_exists = os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    if not db_exists:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            file_path TEXT,
            content TEXT,
            source TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            vector BLOB
        )
        ''')
        conn.commit()
        print(f"Created new database: {db_path}")
    else:
        print(f"Connected to existing database: {db_path}")
    return conn

# Helper Functions
def vectorize_text(text):
    global vectorizer
    vector = vectorizer.fit_transform([text])
    return vector.toarray()[0]

def search_database(query, db_connection):
    global vectorizer
    cursor = db_connection.cursor()
    cursor.execute("SELECT id, title, content, vector FROM content")
    results = cursor.fetchall()
    
    # Fit the vectorizer on all documents (including the query)
    all_texts = [query] + [row[2] for row in results]
    vectorizer.fit(all_texts)
    
    # Transform the query
    query_vector = vectorizer.transform([query]).toarray()[0]
    
    similarities = []
    for row in results:
        db_vector = np.frombuffer(row[3], dtype=np.float64)
        
        # Ensure the db_vector has the same dimensionality as query_vector
        if len(db_vector) != len(query_vector):
            db_vector = vectorizer.transform([row[2]]).toarray()[0]
        
        similarity = cosine_similarity([query_vector], [db_vector])[0][0]
        similarities.append((row[0], row[1], row[2], similarity))
    
    return sorted(similarities, key=lambda x: x[3], reverse=True)

def process_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif ext.lower() == '.html':
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            return soup.get_text()
    elif ext.lower() == '.docx':
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def fetch_url_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        title = page.title()
        browser.close()
    soup = BeautifulSoup(content, 'html.parser')
    return title, soup.get_text()

def create_temp_database(folder_path):
    global vectorizer
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    conn = get_or_create_database(temp_db.name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            file_path TEXT,
            content TEXT,
            source TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            vector BLOB
        )
    ''')
    
    all_content = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            content = process_file(file_path)
            all_content.append(content)
    
    vectorizer.fit(all_content)
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            content = process_file(file_path)
            vector = vectorizer.transform([content]).toarray()[0]
            cursor.execute('''
            INSERT INTO content (title, file_path, content, source, vector)
            VALUES (?, ?, ?, ?, ?)
            ''', (file, file_path, content, 'local_file', vector.tobytes()))
    conn.commit()
    return conn

def is_url(string):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(string) is not None

def format_text_for_display(text):
    paragraphs = text.split('\n')
    return '\n'.join([f"[{i+1}] {para}" for i, para in enumerate(paragraphs) if para.strip()])

def load_file(file_path):
    return process_file(file_path)

def detect_keywords(text, num_keywords):
    if num_keywords == 0:
        return []
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system=f"You are a keyword extraction expert. Extract the {num_keywords} most important keywords from the given text and return them as a JSON list.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract {num_keywords} keywords from this text:\n\n{text}"
                    }
                ]
            }
        ]
    )
    
    # Extract the JSON content from the response
    response_text = message.content[0].text
    json_start = response_text.find('[')
    json_end = response_text.rfind(']') + 1
    
    if json_start != -1 and json_end != -1:
        json_content = response_text[json_start:json_end]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            print("Error parsing JSON response. Using fallback method.")
            # Fallback: split the content by commas and clean up
            keywords = [k.strip().strip('"') for k in json_content.strip('[]').split(',')]
            return keywords
    else:
        print("JSON list not found in response. Using fallback method.")
        # Fallback: split the content by newlines or commas
        keywords = [k.strip() for k in re.split(r'[,\n]', response_text) if k.strip()]
        return keywords[:num_keywords]

def create_writing_plan(topic, search_results, modifications=None):
    context = "\n".join([f"Title: {result[1]}\nContent: {result[2][:500]}...\nSource: {result[3]}" for result in search_results])
    prompt = f"Create a detailed writing plan for an article on the topic: '{topic}'\n\nContext:\n{context}"
    
    if modifications:
        prompt += f"\n\nPlease incorporate the following modifications: {modifications}"
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0,
        system="You are an expert writing planner. Create a detailed writing plan for the given topic using the provided context.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def generate_draft(topic, plan, search_results):
    context = "\n".join([f"Title: {result[1]}\nContent: {result[2][:500]}...\nSource: {result[3]}" for result in search_results])
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        temperature=0.7,
        system="You are an expert writer. Write a comprehensive text matching the request of the user, matching the tone, plan, and context." +
        "Avoid using bullets or lists unless the user specifically requests for that, and use a human persona for writing." +
        "Links or pages, or important quotations you got your information from are available in the context." +
        "If the user asks for citations, don't forget to include this information as needed.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Write an article on '{topic}' using this plan:\n\n{plan}\n\nContext:\n{context}"
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def rewrite_paragraph(original_paragraph, context):
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        system="You are an expert writer. Rewrite the given paragraph while maintaining its original meaning and style." +
        "Avoid using bullets or lists unless the user specifically requests for that, and use a human persona for writing." +
        "Links or pages, or important quotations you got your information from are available in the context." +
        "If the user asks for citations, don't forget to include this information as needed. DO NOT completely change the original paragraph topic.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Rewrite this paragraph:\n\n{original_paragraph}\n\nContext:\n{context}"
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def fact_check(original_paragraph, rewritten_paragraph, search_results):
    context = "\n".join([f"Title: {result[1]}\nContent: {result[2][:500]}...\nSource: {result[3]}" for result in search_results])
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a fact-checking expert. Compare the original and rewritten paragraphs for factual consistency using the provided context.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Compare these paragraphs for factual consistency:\n\nContext:\n{context}\n\nOriginal:\n{original_paragraph}\n\nRewritten:\n{rewritten_paragraph}"
                    }
                ]
            }
        ]
    )

    response_text = message.content[0].text.strip()
    if response_text:
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            print("Error: Response is not in JSON format.")
            result = {'consistent': False, 'issues': []}
    else:
        print("Error: Empty response received.")
        result = {'consistent': False, 'issues': []}

    if not result['consistent']:
        for issue in result['issues']:
            if not validators.url(issue):
                result['issues'].remove(issue)

    return result

def web_search(query, num_results=5, offline=False):
    if offline:
        print("Offline mode: Web search disabled.")
        return []
    
    print(f"Searching the web for: {query}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"https://www.google.com/search?q={query}")
        search_results = page.query_selector_all('.g')
        results = []
        for result in search_results[:num_results]:
            title_elem = result.query_selector('h3')
            link_elem = result.query_selector('a')
            snippet_elem = result.query_selector('.VwiC3b')
            
            if title_elem and link_elem and snippet_elem:
                title = title_elem.inner_text()
                url = link_elem.get_attribute('href')
                snippet = snippet_elem.inner_text()
                results.append((title, url, snippet, "web"))
                print(f"Found: {title} - {url}")
        
        browser.close()
    
    return results

def research_and_write(db_connection, temp_db_connection=None, offline=False):
    topic = input("What do you want to write about? ")
    num_keywords = int(input("How many keywords should we use? (Enter 0 to disable keyword search) "))
    keywords = detect_keywords(topic, num_keywords) if num_keywords > 0 else []
    
    if keywords:
        print("Proposed keywords:")
        print(", ".join(keywords))
    
    while True:
        proceed = input("Should we proceed? ([y]es [n]o [m]anual mode) ").lower()
        if proceed == 'y':
            if num_keywords > 0:
                num_results = int(input("How many results should be added to the context? "))
            break
        elif proceed == 'n':
            num_keywords = int(input("How many keywords should we use? (Enter 0 to disable keyword search) "))
            keywords = detect_keywords(topic, num_keywords) if num_keywords > 0 else []
            if keywords:
                print("Updated keywords:")
                print(", ".join(keywords))
        elif proceed == 'm':
            keywords = []
            print("Enter 1 keyword per line (empty line to finish):")
            while True:
                keyword = input()
                if not keyword:
                    break
                keywords.append(keyword)
            if keywords:
                num_results = int(input("How many results should be added to the context? "))
            break
        else:
            print("Invalid input. Please try again.")
    
    search_results = []
    if keywords and not offline:
        search_results = web_search(' '.join(keywords), num_results, offline)
    
    # Search local database
    offline_results = search_database(topic, db_connection)
    search_results.extend([(result[1], "", result[2], "offline") for result in offline_results[:num_results]])

    # Search temporary database if provided
    if temp_db_connection:
        temp_offline_results = search_database(topic, temp_db_connection)
        search_results.extend([(result[1], "", result[2], "temp_offline") for result in temp_offline_results[:num_results]])

    # Avoid duplicates
    seen_urls = set()
    filtered_results = []
    for result in search_results:
        url = result[1]
        if url and url not in seen_urls:
            seen_urls.add(url)
            filtered_results.append(result)
    search_results = filtered_results

    # Fetch full content for each search result and save to database
    full_results = []
    for title, url, snippet, source in search_results:
        try:
            if url and not offline:
                print(f"Fetching content from: {url}")
                cursor = db_connection.cursor()
                cursor.execute("SELECT content FROM content WHERE url = ?", (url,))
                result = cursor.fetchone()
                if result:
                    full_content = result[0]
                else:
                    full_title, full_content = fetch_url_content(url)
                    vector = vectorize_text(full_content)
                    cursor.execute('''
                    INSERT INTO content (title, url, content, source, vector)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (full_title, url, full_content, 'web', vector.tobytes()))
                    db_connection.commit()
            else:
                full_title, full_content = title, snippet
            full_results.append((full_title, url, full_content, source))
            print(f"Content saved: {full_title}")
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")

    plan = create_writing_plan(topic, full_results)
    print("\nProposed Writing Plan:")
    print(plan)

    while True:
        confirm = input("Is this plan correct? ([y]es / [n]o / [c]ancel) ").lower()
        if confirm == 'y':
            draft = generate_draft(topic, plan, full_results)
            print("\nGenerated Draft:")
            print(draft)

            draft_confirm = input("Is this draft acceptable? ([y]es / [n]o) ").lower()
            if draft_confirm == 'y':
                return draft
            else:
                print("Returning to the writing plan stage.")
        elif confirm == 'n':
            modifications = input("What modifications would you like to make to the plan? ")
            plan = create_writing_plan(topic, full_results, modifications)
            print("\nUpdated Writing Plan:")
            print(plan)
        elif confirm == 'c':
            return None
        else:
            print("Invalid input. Please try again.")
            
def partial_rewrite(text, db_connection, temp_db_connection=None):
    formatted_text = format_text_for_display(text)
    print(formatted_text)

    while True:
        selection = input("Enter the paragraph numbers you want to rewrite (e.g., 2 4-7 9): ")
        paragraphs_to_rewrite = parse_paragraph_selection(selection)

        paragraphs = text.split('\n')
        max_paragraph_number = len(paragraphs)

        for para_num in paragraphs_to_rewrite:
            if para_num < 1 or para_num > max_paragraph_number:
                print(f"Paragraph number {para_num} is out of range. Skipping...")
                continue

            original_paragraph = paragraphs[para_num - 1]
            print(f"\nOriginal paragraph [{para_num}]:")
            print(original_paragraph)

            feedback = input("Do you have any feedback for rewriting this paragraph? (Press Enter if none): ")

            num_keywords = int(input("How many keywords should we use? (Enter 0 to disable keyword search) "))
            keywords = detect_keywords(original_paragraph, num_keywords) if num_keywords > 0 else []

            num_results = 5  # Default value
            while True:
                if keywords:
                    print("Proposed keywords:")
                    print(", ".join(keywords))

                proceed = input("Should we proceed? ([y]es [n]o [m]anual mode) ").lower()
                if proceed == 'y':
                    num_results = int(input("How many results should be added to the context? "))
                    break
                elif proceed == 'n':
                    num_keywords = int(input("How many keywords should we use? (Enter 0 to disable keyword search) "))
                    keywords = detect_keywords(original_paragraph, num_keywords) if num_keywords > 0 else []
                elif proceed == 'm':
                    keywords = []
                    print("Enter 1 keyword per line (empty line to finish):")
                    while True:
                        keyword = input()
                        if not keyword:
                            break
                        keywords.append(keyword)
                    num_results = int(input("How many results should be added to the context? "))
                    break
                else:
                    print("Invalid input. Please try again.")

            search_results = []
            if keywords:
                search_results = web_search(' '.join(keywords), num_results)

            offline_results = search_database(' '.join(keywords), db_connection)
            search_results.extend([(result[1], "", result[2], "offline") for result in offline_results[:num_results]])

            if temp_db_connection:
                temp_offline_results = search_database(' '.join(keywords), temp_db_connection)
                search_results.extend([(result[1], "", result[2], "temp_offline") for result in temp_offline_results[:num_results]])

            context = "\n".join(paragraphs[max(0, para_num-2):min(len(paragraphs), para_num+1)])
            if feedback:
                context += f"\nFeedback: {feedback}"

            # Fetch content from the database using the URLs
            if search_results:
                context += "\nSearch Results:\n"
                for title, url, snippet, source in search_results:
                    cursor = db_connection.cursor()
                    if url:
                        try:
                            cursor.execute("SELECT content FROM content WHERE url = ?", (url,))
                            result = cursor.fetchone()
                            if result:
                                content = result[0]
                            else:
                                content = snippet
                        except:
                            print(f"Error fetching content from URL: {url}")
                            content = snippet
                    else:
                        content = snippet
                    context += f"Title: {title}\nURL: {url}\nContent: {content[:500]}...\nSource: {source}\n"

            rewritten_paragraph = rewrite_paragraph(original_paragraph, context)
            print(f"\nRewritten paragraph [{para_num}]:")
            print(rewritten_paragraph)

            fact_check_result = fact_check(original_paragraph, rewritten_paragraph, search_results)
            if fact_check_result['consistent']:
                print("\nThe rewrite is factually consistent.")
            else:
                print("\nFactual inconsistencies detected:")
                for issue in fact_check_result['issues']:
                    print(f"- {issue}")

            choice = input("Accept this rewrite? ([y]es / [n]o / [e]dit / [s]ources) ").lower()
            if choice == 'y':
                paragraphs[para_num - 1] = rewritten_paragraph
            elif choice == 'e':
                edited_paragraph = input("Enter your edited version of the paragraph:\n")
                paragraphs[para_num - 1] = edited_paragraph
            elif choice == 'n':
                pass
            elif choice == 's':
                paragraphs[para_num - 1] = add_sources_to_paragraph(original_paragraph, db_connection)

        text = '\n'.join(paragraphs)
        print("\nUpdated text:")
        print(format_text_for_display(text))

        more_rewrites = input("Do you want to make any more rewrites? ([y]es / [n]o) ").lower()
        if more_rewrites != 'y':
            break

    return text
    
def add_sources_to_paragraph(text, db_connection):
    global vectorizer  # Ensure we're using the global vectorizer

    paragraphs = text.split('\n')

    while True:
        selection = input("Enter the paragraph number you want to add sources to (or 'q' to quit): ")
        if selection.lower() == 'q':
            break

        try:
            para_num = int(selection)
            if 1 <= para_num <= len(paragraphs):
                selected_paragraph = paragraphs[para_num - 1]
                print(f"\nSelected paragraph:")
                print(selected_paragraph)

                # Search the database for matches to the selected paragraph
                cursor = db_connection.cursor()
                cursor.execute("SELECT title, content, url, vector FROM content")
                db_results = cursor.fetchall()

                # Fit the vectorizer on all content
                all_content = [selected_paragraph] + [row[1] for row in db_results]
                vectorizer.fit(all_content)

                # Get vector for the selected paragraph
                para_vector = vectorizer.transform([selected_paragraph]).toarray()[0]

                # Calculate similarities
                similarities = []
                for row in db_results:
                    db_vector = np.frombuffer(row[3], dtype=np.float64)
                    if len(db_vector) != len(para_vector):
                        db_vector = vectorizer.transform([row[1]]).toarray()[0]
                    similarity = cosine_similarity([para_vector], [db_vector])[0][0]
                    similarities.append((row[2], similarity))  # (url, similarity)

                # Sort matches by similarity score, highest to lowest
                sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)

                print("\nMatches sorted by similarity score (highest to lowest):")
                for i, match in enumerate(sorted_matches, 1):
                    url, score = match
                    print(f"{i}. URL: {url if url else 'Local file'}")
                    print(f"   Similarity score: {score:.4f}")
                    print()

                if sorted_matches:
                    while True:
                        match_selection = input("Enter the number of the match you want to add (or 'skip' to skip): ")
                        if match_selection.lower() == 'skip':
                            break
                        try:
                            match_num = int(match_selection)
                            if 1 <= match_num <= len(sorted_matches):
                                selected_match = sorted_matches[match_num - 1]
                                source_text = selected_match[0] if selected_match[0] else "Local file"
                                paragraphs[para_num - 1] += f" (Source: {source_text}, Similarity: {selected_match[1]:.2f})"
                                print("Source added successfully.")
                                break
                            else:
                                print("Invalid match number. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter a number or 'skip'.")
                else:
                    print("No matching source found for this paragraph.")
            else:
                print("Invalid paragraph number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

    return '\n'.join(paragraphs)
    
def delete_paragraphs(text):
    while True:
        formatted_text = format_text_for_display(text)
        print(formatted_text)

        selection = input("Enter the paragraph numbers you want to delete (e.g., 2 4-7 9): ")
        paragraphs_to_delete = parse_paragraph_selection(selection)

        confirm = input("Are you sure you want to delete these paragraphs? ([y]es / [n]o) ").lower()
        if confirm == 'y':
            paragraphs = text.split('\n')
            text = '\n'.join([para for i, para in enumerate(paragraphs) if i+1 not in paragraphs_to_delete])

        more_deletions = input("Do you want to delete more paragraphs? ([y]es / [n]o) ").lower()
        if more_deletions != 'y':
            break

    return text

def save_content_to_database(db_connection):
    global vectorizer
    source = input("Enter the file path of the content to be saved: ")

    if source.lower().endswith('.db'):
        print("Cannot save SQLite database files. Please choose a different file type.")
        return

    try:
        content = process_file(source)
        title = os.path.basename(source)
        file_path = os.path.abspath(source)
        source_type = 'local_file'
    except FileNotFoundError:
        print(f"Error: File '{source}' not found. Please check the file path and try again.")
        return
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    # Fit the vectorizer on all existing content plus the new content
    cursor = db_connection.cursor()
    cursor.execute("SELECT content FROM content")
    existing_content = [row[0] for row in cursor.fetchall()]
    all_content = existing_content + [content]
    vectorizer.fit(all_content)

    vector = vectorizer.transform([content]).toarray()[0]

    cursor.execute('''
    INSERT INTO content (title, file_path, content, source, vector)
    VALUES (?, ?, ?, ?, ?)
    ''', (title, file_path, content, source_type, vector.tobytes()))
    db_connection.commit()

    print("Content saved successfully.")

def save_draft_to_file(text):
    filename = input("Enter the filename to save the draft: ")
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Draft saved to {filename}")

def parse_paragraph_selection(selection):
    paragraphs = set()
    for item in selection.split():
        if '-' in item:
            start, end = map(int, item.split('-'))
            paragraphs.update(range(start, end+1))
        else:
            paragraphs.add(int(item))
    return paragraphs

def main():
    parser = argparse.ArgumentParser(description="AI Writing Agent")
    parser.add_argument("--content", help="Folder path for content to be vectorized")
    parser.add_argument("--file", help="File path to load initial content")
    parser.add_argument("--db", default="content.db", help="Path to the SQLite database (default: content.db)")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (no web searches)")
    args = parser.parse_args()

    db_connection = get_or_create_database(args.db)

    if args.content:
        temp_db_connection = create_temp_database(args.content)
    else:
        temp_db_connection = None

    current_text = ""
    if args.file:
        current_text = load_file(args.file)

    try:
        while True:
            print("\nMain Menu:")
            if current_text:
                print(format_text_for_display(current_text))
            print("1. Research and write")
            print("2. Partial rewrite")
            print("3. Delete paragraph(s)")
            print("4. Save content references into database")
            print("5. Add sources to paragraph")
            print("6. Save draft")
            print("7. Exit")

            choice = input("Enter your choice (1-7): ")

            # Clear any previous AI responses
            current_text = current_text.split('\n\nMain Menu:')[0]

            if choice == '1':
                new_text = research_and_write(db_connection, temp_db_connection, args.offline)
                if new_text:
                    current_text = new_text
            elif choice == '2':
                current_text = partial_rewrite(current_text, db_connection, temp_db_connection)
            elif choice == '3':
                current_text = delete_paragraphs(current_text)
            elif choice == '4':
                save_content_to_database(db_connection)
            elif choice == '5':
                current_text = add_sources_to_paragraph(current_text, db_connection)
            elif choice == '6':
                if current_text:
                    save_draft_to_file(current_text)
                else:
                    print("No draft to save. Please research and write or load a file first.")
            elif choice == '7':
                break
            else:
                print("Invalid choice. Please try again.")

    finally:
        db_connection.close()
        if temp_db_connection:
            temp_db_connection.close()

if __name__ == "__main__":
    main()