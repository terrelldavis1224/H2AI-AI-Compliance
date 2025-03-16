# H2AI-AI-Compliance

CompliTracker

CompliTracker is a web application designed to help hospitals identify and correct errors in forms before submission, reducing rejection rates and improving efficiency.

Features

File Upload: Users can upload JSON files containing error data.

Table Display: The application dynamically displays errors with their corresponding probabilities.

Local Storage: Data persists in the browser's local storage for session continuity.

Clear Functionality: Users can clear the table and stored data with a single button click.

Responsive Design: Built with Tailwind CSS for a modern and responsive user interface.

Technologies Used

HTML for structuring the page.

CSS (Tailwind CSS) for styling and layout.

JavaScript for handling file uploads, table population, and local storage.

Installation and Setup

Clone the repository:

git clone https://github.com/your-repo/complitracker.git

Navigate to the project directory:

cd complitracker

Open the index.html file in a web browser to run the application.

Usage

Click on the "Choose File" button to upload a JSON file.

The uploaded file should contain an array of objects with the following structure:


```[
  {
    "id": 1,
    "errors": ["Missing Date", "Incorrect Format"],
    "probability": 0.85
  },
  {
    "id": 2,
    "errors": ["Incomplete Fields"],
    "probability": 0.72
  }
]
```
The table updates to display the errors and their probability of causing rejection.

Click the "Clear" button to remove data from the table and local storage.
```
Project Structure

complitracker/
│── index.html        # Main HTML file
│── script.js         # JavaScript for file handling and UI updates
│── styles.css        # Optional additional styles
└── README.md         # Project documentation
└── model             # Machine learning code

```
Future Enhancements

Implement form validation before uploading JSON files.

Enhance error visualization using charts or graphs.

Add backend integration for database storage.

Contributors

Sayan Pelrecha

Hetu Chauhan

Pragalv Bhattarai

Terrell Davis

Raymond Quan

License

This project is licensed under the MIT License.

