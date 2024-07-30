
// IMPORT ALL NEEDED CONSTANTS
const dropArea = document.getElementById('dropArea');
const dropText = document.getElementById('dropText');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');


// IMPORT ALL NEEDED VARIABLES
var alertDiv = document.getElementById('alertDiv');
var outputBlock = document.getElementById('outputBlock');
var outputMessage = document.getElementById('outputMessage');
var imgElement = document.getElementById("responseImage");

// GLOBAL VARIABLE FOR WHICH CONTAINS LAST UPLOADED IMAGE
let file;



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

// Function to show the outputMessage with fadeIn effect
function showOutputBlock(outputBlock) {
    outputBlock.style.opacity = '1';
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
file = e.dataTransfer.files[0];
console.log(file);
displayImage(file);
showAlert(alertDiv);
});

dropArea.addEventListener('click', () => {
fileInput.click();
});

fileInput.addEventListener('change', () => {
file = fileInput.files[0];
console.log(file);
displayImage(file);
showAlert(alertDiv);
});

document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();

    var fileInput = document.getElementById('fileInput');
    var alertDiv = document.getElementById('alertDiv');
    var previewImage = document.getElementById('previewImage');
    var dropArea = document.getElementById('dropArea');

    console.log("Ohooooj", file);

    // Check if a file is selected
    if (file != undefined) {
        var formData = new FormData();
        formData.append('file', file);



        // Use Fetch API to send the image to the server
        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())  // Parse the JSON response
        .then(result => {
            // Handle the result from the server
            console.log(result);

            // UPDATE YOUR UI OR PERFORM ANY OTHER ACTIONS BASED ON THE RESULT
            // GET SERVER ROUTE RESULT
            // Access the predicted_class and predicted_probability
            const predictedClass = result.result.predicted_class;
            const predictedProbability = result.result.predicted_probability;

            if (predictedClass === 0) {

            // Change the inner text
            outputMessage.innerText = 'Good news! This is not a tumor! I am saying it with '+  predictedProbability + ' confidence.';
            showOutputBlock(outputBlock)

            }

            else if (predictedClass === 1) {

            // Change the inner text
            outputMessage.innerText = 'Unfortunately, this looks as a tumor. I am saying it with '+  predictedProbability + ' confidence. Be strong and consult your specialist doctor regarding the following steps. You will win this fight, I truly believe in you.';
            showOutputBlock(outputBlock)

            }

            imgElement.src = "static/1_doctorsImages";


            //showAlert(alertDiv);
        })
        .catch(error => console.error('Error:', error));
    } else {
        // Handle the case where no file is selected
        imgElement.src = "static/2_serviceOperaterImages/serviceOperator.png";
        outputMessage.innerText = 'Jes retardiran, dodaj sliku.';
        showOutputBlock(outputBlock)
    }
});