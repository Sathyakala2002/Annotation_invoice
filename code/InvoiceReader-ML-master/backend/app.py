from flask import Flask, request
import Train_Model 

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_url = data.get('image_url')
    Train_Model.process_single_invoice_image(image_url)
    return {'message': 'Image processed successfully'}

if __name__ == '__main__':
    app.run(debug=True)
