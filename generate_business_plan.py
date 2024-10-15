import sys
import json
import requests
import zipfile
import os
from io import BytesIO
from PIL import Image
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QMessageBox, QProgressBar, QInputDialog
from PyQt5.QtCore import QThread, pyqtSignal

# OpenAI and DALL-E setup
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
API_KEY_FILE = "api_key.json"

HEADERS = {
    "Authorization": "",
    "Content-Type": "application/json"
}

class QuickActionThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bytes, str)

    def __init__(self, action, prompt, parent=None):
        super().__init__(parent)
        self.action = action
        self.prompt = prompt

    def run(self):
        try:
            if self.action == "business plan":
                result = self.generate_business_plan()
            else:
                result = self.generate_content(self.prompt)

            if isinstance(result, dict):
                zip_data = self.create_zip(result)
                self.finished.emit(zip_data, f"{self.action}.zip")
            else:
                self.finished.emit(None, result)
        except Exception as e:
            self.finished.emit(None, f"Error: {str(e)}")

    def generate_content(self, prompt):
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant specializing in {self.action}."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(CHAT_API_URL, headers=HEADERS, json=data)
            response.raise_for_status()
            response_data = response.json()
            if "choices" not in response_data:
                error_message = response_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}"

            content_text = response_data["choices"][0]["message"]["content"]
            return content_text

        except requests.RequestException as e:
            return f"Error: Unable to communicate with the OpenAI API."

    def generate_business_plan(self):
        business_plan = {}
        user_prompt = self.prompt
        try:
            self.progress.emit(10, "Generating business concept...")
            business_concept = self.generate_content(f"Create a detailed business concept based on the following prompt: {user_prompt}.")
            business_plan['business_concept'] = business_concept

            self.progress.emit(20, "Generating market analysis...")
            business_plan['market_analysis'] = self.generate_content(f"Create a detailed market analysis for the business: {business_concept}")

            self.progress.emit(30, "Generating SWOT analysis...")
            business_plan['swot_analysis'] = self.generate_content(f"Create a SWOT analysis for the business: {business_concept}")

            self.progress.emit(40, "Generating marketing strategy...")
            business_plan['marketing_strategy'] = self.generate_content(f"Create a detailed marketing strategy for the business: {business_concept}")

            self.progress.emit(50, "Generating financial projections...")
            business_plan['financial_projections'] = self.generate_financial_projections()

            self.progress.emit(60, "Generating executive summary...")
            business_plan['executive_summary'] = self.generate_content(f"Create an executive summary for the business plan: {business_concept}")

            self.progress.emit(70, "Generating recap...")
            business_plan['recap'] = self.generate_content(f"Recap the business plan: {business_concept}")

            self.progress.emit(80, "Generating master document...")
            business_plan['master_document'] = self.create_master_document(business_plan)

            self.progress.emit(90, "Packaging into ZIP...")
            return business_plan

        except Exception as e:
            return f"Error during business plan generation: {str(e)}"

    def generate_financial_projections(self):
        try:
            profit_loss_data = {
                "Description": ["Revenue", "Cost of Goods Sold", "Gross Profit", "Operating Expenses", "Net Profit"],
                "Amount": [50000, 20000, 30000, 15000, 15000]
            }
            cash_flow_data = {
                "Description": ["Cash Inflow", "Cash Outflow", "Net Cash Flow"],
                "Amount": [60000, 30000, 30000]
            }
            balance_sheet_data = {
                "Description": ["Assets", "Liabilities", "Equity"],
                "Amount": [70000, 40000, 30000]
            }
            projections_bytes = BytesIO()
            with pd.ExcelWriter(projections_bytes, engine='xlsxwriter') as writer:
                pd.DataFrame(profit_loss_data).to_excel(writer, index=False, sheet_name='Profit and Loss')
                pd.DataFrame(cash_flow_data).to_excel(writer, index=False, sheet_name='Cash Flow')
                pd.DataFrame(balance_sheet_data).to_excel(writer, index=False, sheet_name='Balance Sheet')
            projections_bytes.seek(0)
            return projections_bytes.read()
        except Exception as e:
            print(f"Error generating financial projections: {str(e)}")
            return None

    def create_master_document(self, business_plan):
        master_doc = "Business Plan Master Document\n\n"
        for key, value in business_plan.items():
            if key == "financial_projections":
                master_doc += f"{key.replace('_', ' ').capitalize()}: See attached Excel file.\n"
            else:
                master_doc += f"{key.replace('_', ' ').capitalize()}: See attached document.\n"
        return master_doc

    def create_zip(self, content_dict):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for key, value in content_dict.items():
                if isinstance(value, str):
                    zip_file.writestr(f"{key}.txt", value)
                elif isinstance(value, bytes):
                    zip_file.writestr(key, value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str):
                            zip_file.writestr(f"{key}/{sub_key}.txt", sub_value)
                        elif isinstance(sub_value, bytes):
                            zip_file.writestr(f"{key}/{sub_key}", sub_value)

        zip_buffer.seek(0)
        self.progress.emit(100, "ZIP package created.")
        return zip_buffer.read()

class QuickActionsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quick Actions - Business Plan Generator")

        self.api_key = self.load_api_key()
        if not self.api_key:
            self.api_key = self.ask_api_key()
            if not self.api_key:
                QMessageBox.critical(self, "Error", "API key is required to proceed.")
                sys.exit()

        global HEADERS
        HEADERS = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Main layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.prompt_label = QLabel("Enter topic/keywords:")
        self.prompt_entry = QLineEdit()
        self.main_layout.addWidget(self.prompt_label)
        self.main_layout.addWidget(self.prompt_entry)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.main_layout.addWidget(self.result_box)

        self.progress_bar = QProgressBar()
        self.main_layout.addWidget(self.progress_bar)

        self.actions = [
            "business plan"
        ]

        for action in self.actions:
            button = QPushButton(f"Generate {action.capitalize()}")
            button.clicked.connect(lambda checked, a=action: self.handle_action(a))
            self.main_layout.addWidget(button)

    def load_api_key(self):
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, 'r') as file:
                data = json.load(file)
                return data.get('api_key')
        return None

    def ask_api_key(self):
        api_key, ok = QInputDialog.getText(self, "API Key", "Please enter your OpenAI API key:", QLineEdit.Password)
        if ok:
            with open(API_KEY_FILE, 'w') as file:
                json.dump({"api_key": api_key}, file)
            return api_key
        return None

    def handle_action(self, action):
        prompt = self.prompt_entry.text()
        self.result_box.append(f"Generating {action}...")
        self.progress_bar.setValue(0)
        self.quick_action_thread = QuickActionThread(action, prompt)
        self.quick_action_thread.progress.connect(self.update_progress)
        self.quick_action_thread.finished.connect(self.handle_finished)
        self.quick_action_thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.result_box.append(message)

    def handle_finished(self, zip_data, filename_or_error):
        if zip_data:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save ZIP", "", "Zip Files (*.zip);;All Files (*)", options=options)
            if file_path:
                with open(file_path, 'wb') as file:
                    file.write(zip_data)
            self.result_box.append(f"{filename_or_error} generated and saved.")
        else:
            self.result_box.append(filename_or_error)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuickActionsApp()
    window.show()
    sys.exit(app.exec_())
