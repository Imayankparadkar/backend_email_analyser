import os
import json
import csv
import io
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AIBusinessAdvisor:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.groq_base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Debug print
        print(f"GROQ API Key loaded: {'Yes' if self.groq_api_key else 'No'}")
        if self.groq_api_key:
            print(f"GROQ API Key starts with: {self.groq_api_key[:10]}...")
        
    def test_connection(self):
        """Test GROQ API connection"""
        if not self.groq_api_key:
            return False, "GROQ API key not found in environment variables"
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test payload
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            print("Testing GROQ API connection...")
            response = requests.post(self.groq_base_url, headers=headers, json=payload, timeout=30)
            print(f"GROQ Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                return True, "GROQ API connected successfully"
            else:
                print(f"GROQ Error Response: {response.text}")
                return False, f"GROQ API error: {response.status_code} - {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return False, "GROQ API connection timeout"
        except requests.exceptions.RequestException as e:
            return False, f"GROQ API connection error: {str(e)}"
        except Exception as e:
            return False, f"GROQ API unexpected error: {str(e)}"
        
    def analyze_data(self, data, role='CEO', user_context=None):
        """Analyze business data using GROQ API and provide strategic insights"""
        
        if not self.groq_api_key:
            return "Error: GROQ API key not configured. Please check your .env file."
        
        # Role-specific prompts
        role_prompts = {
            'CEO': "You are a strategic business advisor for a CEO. Focus on high-level insights, growth opportunities, risk management, and executive decision-making.",
            'Marketer': "You are a marketing strategist. Focus on campaign performance, customer acquisition, retention strategies, and marketing ROI optimization.",
            'Analyst': "You are a business analyst. Focus on data trends, statistical insights, performance metrics analysis, and detailed recommendations."
        }
        
        # Market trends data (simulated - in production, this could be fetched from external APIs)
        market_trends = {
            "industry_growth": "7.2% YoY growth in digital services sector",
            "seasonal_trends": "Q4 typically shows 15-20% higher conversion rates",
            "market_conditions": "Current market shows strong demand for AI-powered solutions",
            "competitive_landscape": "Increased competition in SaaS space, focus on differentiation needed"
        }
        
        # Prepare the prompt
        system_prompt = f"""
        {role_prompts.get(role, role_prompts['CEO'])}
        
        Current Market Context:
        - Industry Growth: {market_trends['industry_growth']}
        - Seasonal Trends: {market_trends['seasonal_trends']}
        - Market Conditions: {market_trends['market_conditions']}
        - Competitive Landscape: {market_trends['competitive_landscape']}
        
        Analyze the provided business data and give actionable, strategic recommendations. 
        Be specific, data-driven, and focus on ROI impact. Structure your response with:
        1. Key Insights (3-4 bullet points)
        2. Strategic Recommendations (3-5 actionable items)
        3. Risk Alerts (if any)
        4. Next Steps (priority actions for next 30 days)
        
        Make recommendations human-like and strategic, not just data summaries.
        """
        
        user_prompt = f"""
        Business Data Analysis:
        {json.dumps(data, indent=2)}
        
        Additional Context: {user_context or 'No additional context provided'}
        
        Please provide strategic business advice based on this data.
        """
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            print("Sending request to GROQ API...")
            response = requests.post(self.groq_base_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"GROQ API Error: {response.status_code} - {response.text}")
                return f"Error analyzing data: GROQ API returned {response.status_code}. Please check your API key and try again."
            
        except requests.exceptions.RequestException as e:
            print(f"GROQ Request Error: {str(e)}")
            return f"Error analyzing data: {str(e)}"
        except Exception as e:
            print(f"GROQ Unexpected Error: {str(e)}")
            return f"Unexpected error: {str(e)}"

class EmailService:
    def __init__(self):
        # Load Resend API configuration
        self.resend_api_key = os.getenv('RESEND_API_KEY')
        self.resend_api_url = "https://api.resend.com/emails"
        self.resend_from_email = os.getenv('RESEND_FROM_EMAIL')
        self.resend_from_name = os.getenv('RESEND_FROM_NAME', 'AI Business Advisor')
        
        # Debug print
        print(f"Resend API Key loaded: {'Yes' if self.resend_api_key else 'No'}")
        if self.resend_api_key:
            print(f"Resend API Key starts with: {self.resend_api_key[:10]}...")
        print(f"Resend From Email: {self.resend_from_email}")
        
    def get_troubleshooting_info(self):
        """Get troubleshooting information for Resend API setup"""
        return """
        Resend API Setup Instructions:
        
        1. Get your API key from Resend:
           - Login to https://resend.com/
           - Go to API Keys
           - Create a new API key
        
        2. Your .env file should look like:
           RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxxx
           RESEND_FROM_EMAIL=your-verified-sender@domain.com
           RESEND_FROM_NAME=Your Business Name
        
        3. Common issues:
           - API key format incorrect (should start with 're_')
           - Sender email domain not verified in Resend
           - Daily sending limit reached
           - Account suspended or limited
        
        4. Verify sender domain:
           - Go to Domains in Resend dashboard
           - Add and verify your domain
           - Use an email from the verified domain as RESEND_FROM_EMAIL
        """
        
    def test_connection(self):
        """Test Resend API connection"""
        if not self.resend_api_key:
            return False, f"Resend API key not found in environment variables.\n\n{self.get_troubleshooting_info()}"
        
        if not self.resend_from_email:
            return False, f"Resend from email not configured.\n\n{self.get_troubleshooting_info()}"
        
        headers = {
            'Authorization': f'Bearer {self.resend_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Test with a simple email validation (dry run)
        test_payload = {
            "from": f"{self.resend_from_name} <{self.resend_from_email}>",
            "to": ["test@example.com"],  # This won't actually send
            "subject": "Test Connection",
            "text": "This is a test connection email."
        }
        
        try:
            print("Testing Resend API connection...")
            # First, let's try to get API key info if available
            response = requests.get('https://api.resend.com/domains', headers=headers, timeout=10)
            print(f"Resend Response Status: {response.status_code}")
            
            if response.status_code == 200:
                domains = response.json()
                domain_count = len(domains.get('data', []))
                return True, f"✓ Resend API connected successfully. Verified domains: {domain_count}"
            elif response.status_code == 401:
                return False, f"Resend API authentication failed: Invalid API key\n\n{self.get_troubleshooting_info()}"
            else:
                print(f"Resend Error Response: {response.text}")
                return False, f"Resend API error: {response.status_code} - {response.text[:200]}\n\n{self.get_troubleshooting_info()}"
                
        except requests.exceptions.Timeout:
            return False, "Resend API connection timeout"
        except requests.exceptions.RequestException as e:
            return False, f"Resend API connection error: {str(e)}"
        except Exception as e:
            return False, f"Resend API unexpected error: {str(e)}"
    
    def generate_pdf_report(self, content, filename):
        """Generate a PDF report from the content"""
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            
            # Create PDF document
            doc = SimpleDocTemplate(temp_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            title = Paragraph("Strategic Business Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Add date
            date_para = Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal'])
            story.append(date_para)
            story.append(Spacer(1, 12))
            
            # Split content into paragraphs and add to PDF
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Clean up the paragraph text for PDF
                    clean_para = para.strip().replace('\n', ' ')
                    p = Paragraph(clean_para, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            return temp_path
            
        except Exception as e:
            print(f"PDF generation error: {str(e)}")
            return None
    
    def send_report(self, to_email, subject, content, include_pdf=False, user_name="User"):
        """Send strategic report via Resend API"""
        if not self.resend_api_key or not self.resend_from_email:
            return False, f"Resend API credentials not configured.\n\n{self.get_troubleshooting_info()}"
        
        headers = {
            'Authorization': f'Bearer {self.resend_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Create HTML content
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                    Strategic Business Report
                </h1>
                <p style="font-size: 16px; color: #666;">
                    Hello {user_name},<br><br>
                    Your AI-powered business analysis is ready. Here are the key insights and recommendations:
                </p>
                <div style="background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 20px; margin: 20px 0;">
                    {content.replace(chr(10), '<br>')}
                </div>
                <div style="margin-top: 30px; padding: 20px; background-color: #e8f4f8; border-radius: 8px;">
                    <h3 style="color: #2c3e50; margin-top: 0;">About This Report</h3>
                    <p style="margin-bottom: 0; color: #666;">
                        This report was generated using AI analysis of your business data, combined with current market trends and industry insights. The recommendations are tailored for strategic decision-making and ROI optimization.
                    </p>
                </div>
                <div style="margin-top: 30px; text-align: center; color: #888; font-size: 14px;">
                    <p>Generated by AI-Powered Strategic Email Assistant</p>
                    <p>Report Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p>Sent via Resend API</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare email payload
        email_data = {
            "from": f"{self.resend_from_name} <{self.resend_from_email}>",
            "to": [to_email],
            "subject": subject,
            "html": html_content,
            "text": content
        }
        
        # Add PDF attachment if requested
        if include_pdf:
            try:
                pdf_path = self.generate_pdf_report(content, "strategic_report.pdf")
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_content = base64.b64encode(pdf_file.read()).decode()
                    
                    email_data["attachments"] = [{
                        "filename": f"Strategic_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        "content": pdf_content,
                        "content_type": "application/pdf"
                    }]
                    
                    # Clean up temp file
                    os.unlink(pdf_path)
                    print("PDF attachment added to Resend email")
                else:
                    print("PDF generation failed, sending email without attachment")
            except Exception as pdf_error:
                print(f"PDF attachment error: {str(pdf_error)}")
                print("Continuing without PDF attachment...")
        
        try:
            print("Sending email via Resend API...")
            print(f"To: {to_email}")
            print(f"From: {self.resend_from_email}")
            print(f"Subject: {subject}")
            
            response = requests.post(self.resend_api_url, headers=headers, json=email_data, timeout=30)
            print(f"Resend API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                message_id = result.get('id', 'Unknown')
                return True, f"✓ Email sent successfully via Resend API (Message ID: {message_id})"
            else:
                print(f"Resend API Error Response: {response.text}")
                try:
                    error_detail = response.json()
                    error_message = error_detail.get('message', 'Unknown error')
                    return False, f"Resend API error {response.status_code}: {error_message}"
                except:
                    return False, f"Resend API error {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            return False, "Resend API request timeout"
        except requests.exceptions.RequestException as e:
            return False, f"Resend API request error: {str(e)}"
        except Exception as e:
            print(f"Resend API unexpected error: {str(e)}")
            print(traceback.format_exc())
            return False, f"Resend API unexpected error: {str(e)}"

# Initialize services
ai_advisor = AIBusinessAdvisor()
email_service = EmailService()

def parse_csv_data(file_path):
    """Parse CSV file and extract business insights"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic data analysis
        analysis = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "data_sample": df.head(5).to_dict('records'),
            "summary_stats": {}
        }
        
        # Calculate summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            analysis["summary_stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else 0,
                "sum": float(df[col].sum()) if not df[col].isna().all() else 0,
                "min": float(df[col].min()) if not df[col].isna().all() else 0,
                "max": float(df[col].max()) if not df[col].isna().all() else 0
            }
        
        # Look for common business metrics
        revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'income' in col.lower()]
        cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'expense' in col.lower()]
        user_cols = [col for col in df.columns if 'user' in col.lower() or 'customer' in col.lower()]
        
        if revenue_cols:
            analysis["revenue_insights"] = {
                "total_revenue": float(df[revenue_cols[0]].sum()),
                "avg_revenue": float(df[revenue_cols[0]].mean()),
                "revenue_trend": "increasing" if df[revenue_cols[0]].diff().mean() > 0 else "decreasing"
            }
        
        if cost_cols:
            analysis["cost_insights"] = {
                "total_costs": float(df[cost_cols[0]].sum()),
                "avg_costs": float(df[cost_cols[0]].mean())
            }
        
        if user_cols:
            analysis["user_insights"] = {
                "total_users": float(df[user_cols[0]].sum()) if df[user_cols[0]].dtype in ['int64', 'float64'] else len(df[user_cols[0]].unique()),
                "unique_users": len(df[user_cols[0]].unique())
            }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Error parsing CSV: {str(e)}"}

def parse_json_data(file_path):
    """Parse JSON file and extract business insights"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            "data_type": "json",
            "structure": type(data).__name__,
            "content": data if isinstance(data, dict) else {"data": data},
            "total_records": len(data) if isinstance(data, list) else 1
        }
        
    except Exception as e:
        return {"error": f"Error parsing JSON: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get form data
        email = request.form.get('email')
        role = request.form.get('role', 'CEO')
        context = request.form.get('context', '')
        include_pdf = request.form.get('include_pdf') == 'true'
        user_name = request.form.get('user_name', 'User')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Parse data based on file type
        if filename.lower().endswith('.csv'):
            data = parse_csv_data(file_path)
        elif filename.lower().endswith('.json'):
            data = parse_json_data(file_path)
        else:
            return jsonify({"error": "Unsupported file type. Please upload CSV or JSON files."}), 400
        
        if "error" in data:
            return jsonify(data), 400
        
        # Generate AI insights
        ai_analysis = ai_advisor.analyze_data(data, role, context)
        
        # Send email report
        subject = f"Strategic Business Report - {role} Analysis ({datetime.now().strftime('%Y-%m-%d')})"
        success, message = email_service.send_report(
            email, subject, ai_analysis, include_pdf, user_name
        )
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Analysis complete! Strategic report sent to your email.",
                "preview": ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis
            })
        else:
            return jsonify({"error": f"Analysis completed but email failed: {message}"}), 500
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/test-connection', methods=['POST'])
def test_connection():
    """Test API connections"""
    print("Testing connections...")
    results = {}
    
    # Test GROQ API
    try:
        groq_success, groq_message = ai_advisor.test_connection()
        results['groq'] = {
            "status": "success" if groq_success else "error", 
            "message": groq_message
        }
    except Exception as e:
        print(f"GROQ test error: {str(e)}")
        results['groq'] = {"status": "error", "message": f"GROQ test failed: {str(e)}"}
    
    # Test Resend API
    try:
        email_success, email_message = email_service.test_connection()
        results['resend'] = {
            "status": "success" if email_success else "error", 
            "message": email_message
        }
    except Exception as e:
        print(f"Resend test error: {str(e)}")
        results['resend'] = {"status": "error", "message": f"Resend test failed: {str(e)}"}
    
    print(f"Test results: {results}")
    return jsonify(results)

if __name__ == '__main__':
    print("=" * 50)
    print("AI-Powered Strategic Email Assistant")
    print("=" * 50)
    print(f"Environment variables status:")
    print(f"  GROQ_API_KEY: {'✓ Loaded' if os.getenv('GROQ_API_KEY') else '✗ Missing'}")
    print(f"  RESEND_API_KEY: {'✓ Loaded' if os.getenv('RESEND_API_KEY') else '✗ Missing'}")
    print(f"  RESEND_FROM_EMAIL: {'✓ Loaded' if os.getenv('RESEND_FROM_EMAIL') else '✗ Missing'}")
    print(f"  RESEND_FROM_NAME: {'✓ Loaded' if os.getenv('RESEND_FROM_NAME') else '✗ Missing (will use default)'}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)