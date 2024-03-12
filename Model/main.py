# main.py
from predict_cleanliness import predict_cleanliness
from update_sheets import update_google_sheets

model = load_model('cleanliness_model.h5')
image_path = 'path/to/desk_image.jpg'
cleanliness_score = predict_cleanliness(image_path, model)

spreadsheet_id = 'your_spreadsheet_id'
sheet_name = 'Sheet1'
update_google_sheets(spreadsheet_id, sheet_name, cleanliness_score)
