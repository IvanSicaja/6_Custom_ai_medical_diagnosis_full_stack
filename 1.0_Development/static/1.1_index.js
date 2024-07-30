
// IMPORT ALL NEEDED CONSTANTS
const dropArea = document.getElementById('dropArea');
const dropText = document.getElementById('dropText');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');


// IMPORT ALL NEEDED VARIABLES
var alertDiv = document.getElementById('alertDiv');


//FUNCTIONS DEFINITIONS
// Function to show the alert with fadeIn effect
function showAlert(alertDiv) {
    alertDiv.style.opacity = '1';

    // Check if opacity is equal to '1'
    if (alertDiv.style.opacity === '1') {
        // Trigger hideAlert after 2 seconds (2000 milliseconds)
        setTimeout(function () {
            hideAlert(alertDiv);
        }, 2000);
    }
}

// Function to hide the alert with fadeOut effect
function hideAlert(alertDiv) {
    alertDiv.style.opacity = '0';
}

function displayImage(file) {
if (file) {
const reader = new FileReader();
reader.onload = (e) => {
previewImage.src = e.target.result;
previewImage.style.display = 'block';
};
reader.readAsDataURL(file);
// Remove the element from the DOM
dropText.style.display = 'none';
}
}

// DRAG AND DROP EVENTLISTENERS DEFINITIONS
dropArea.addEventListener('dragover', (e) => {
e.preventDefault();
dropText.innerText = 'Release your image now.';
dropArea.style.backgroundColor = 'rgb(68, 70, 84)';

});

dropArea.addEventListener('dragleave', () => {
dropText.innerText = 'Drag and drop your image here or click to browse.';
dropArea.style.backgroundColor = 'rgb(52, 53, 65)';
});

dropArea.addEventListener('drop', (e) => {
e.preventDefault();
const file = e.dataTransfer.files[0];
displayImage(file);
showAlert(alertDiv);
});

dropArea.addEventListener('click', () => {
fileInput.click();
});

fileInput.addEventListener('change', () => {
const file = fileInput.files[0];
displayImage(file);
showAlert(alertDiv);
});

document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();

    var fileInput = document.getElementById('fileInput');
    var alertDiv = document.getElementById('alertDiv');
    var previewImage = document.getElementById('previewImage');

    // Check if a file is selected
    if (fileInput.files.length > 0) {
        var formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Use Fetch API to send the image to the server
        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())  // Parse the JSON response
        .then(result => {
            // Handle the result from the server
            console.log(result);
            // Update your UI or perform any other actions based on the result
            showAlert(alertDiv);
        })
        .catch(error => console.error('Error:', error));
    } else {
        // Handle the case where no file is selected
    }
});