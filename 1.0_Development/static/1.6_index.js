
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
// Get the elements with the ids "customSwitch1", "customSwitch2", and "customSwitch3"
var switch1 = document.getElementById("customSwitch1");
var switch2 = document.getElementById("customSwitch2");
var switch3 = document.getElementById("customSwitch3");
switch1.checked = true;
// Get the target element to scroll to (e.g., outputBlock)
var targetElement = document.getElementById("outputBlock");
// Get the arrowGoOnTheTop element
var arrowGoOnTheTop = document.getElementById('arrowGoOnTheTop');
// Get the target element you want to scroll to (e.g., #navBar)
var targetElementNavBar = document.getElementById('navBar');


// GLOBAL VARIABLE FOR WHICH CONTAINS LAST UPLOADED IMAGE
let file;



//TEMP

//

// Define a function to handle the change event
function handleCheckboxChange(event) {
var checkbox = event.target;

// Check which switch triggered the event
if (checkbox === switch1 && checkbox.checked) {
  // If switch1 is checked, uncheck switch2 and switch3
  switch2.checked = false;
  switch3.checked = false;
} else if (checkbox === switch2 && checkbox.checked) {
  // If switch2 is checked, uncheck switch1 and switch3
  switch1.checked = false;
  switch3.checked = false;
} else if (checkbox === switch3 && checkbox.checked) {
  // If switch3 is checked, uncheck switch1 and switch2
  switch1.checked = false;
  switch2.checked = false;
}

// Add your additional logic here based on the checkbox state
// For example, you might want to perform some action when a checkbox is checked or unchecked.
}

// Add the event listener to switch1
if (switch1) {
switch1.addEventListener("change", handleCheckboxChange);
}

// Add the event listener to switch2
if (switch2) {
switch2.addEventListener("change", handleCheckboxChange);
}

// Add the event listener to switch3
if (switch3) {
switch3.addEventListener("change", handleCheckboxChange);
}


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

// Function to show the outputBlock with fadeIn effect
function showOutputBlock(outputBlock) {
    outputBlock.style.transition = "opacity 1s ease";
    outputBlock.style.opacity = '1';
}

// Function to hide the outputBlock immediately
function hideOutputBlock(outputBlock) {
    outputBlock.style.transition = "none"; // Remove transition effect
    outputBlock.style.opacity = '0';
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

// ADD A CLICK EVENT LISTENER TO THE ARROWGOONTHETOP ELEMENT
arrowGoOnTheTop.addEventListener('click', function (event) {
    event.preventDefault(); // Prevent the default behavior of the link

    // Scroll to the target element with smooth behavior
    targetElementNavBar.scrollIntoView({
        behavior: 'smooth'
    });
});


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

    // Function which hides output block
    hideOutputBlock(outputBlock)

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
            // Select random output image
            // Access the wanted key from the server result
            var outputImageNames = result.result.responseImageNames


            // Check if the element is found
            if (imgElement && outputImageNames.length > 0) {
            // Generate a random index to select an image
            var randomIndex = Math.floor(Math.random() * outputImageNames.length);

            // Construct the path to the randomly selected image
            var randomImagePath = "static/1_doctorsImages/" + outputImageNames[randomIndex];

            // Create a new Image object in order to know whae the image is fully uploaded in cache memory
            var tempImage = new Image();

            // Set the new source for the image
            tempImage.src = randomImagePath;

            // Set the onload event handler
            tempImage.onload = function () {
                // This code will be executed once the image is fully loaded
                // Set the new source for the imgElement
                imgElement.src = randomImagePath;

                // Any additional code you want to execute after the image is loaded
                console.log("Image is fully loaded!");

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
            };

            // Note: If the image is already in the browser cache, the onload event may trigger immediately
            // To handle this case, you can check if the image is complete before setting the onload event
            if (tempImage.complete) {
                tempImage.onload(); // Manually trigger the onload event
            }

            }
            else {
            console.error("Element with ID 'responseImage' not found or no images available.");
            }

            //SCROLL TO THE RESPONSE MESSAGE
            // Scroll smoothly to the target element
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: "smooth" });
            }
            //

        })
        .catch(error => console.error('Error:', error));
    }

    else {
        // Handle the case where no file is selected
        // Create a new Image object in order to know whae the image is fully uploaded in cache memory
        var tempImage = new Image();

        // Set the new source for the image
        tempImage.src = "static/2_serviceOperaterImages/serviceOperator.png";

        // Set the onload event handler
        tempImage.onload = function () {
            // This code will be executed once the image is fully loaded
            // Set the new source for the imgElement
            imgElement.src = "static/2_serviceOperaterImages/serviceOperator.png";

            // Any additional code you want to execute after the image is loaded
            console.log("Image is fully loaded!");

            // Define output message and show the output block
            outputMessage.innerText = 'Jes retardiran, dodaj sliku.';
            showOutputBlock(outputBlock)
        };

        // Note: If the image is already in the browser cache, the onload event may trigger immediately
        // To handle this case, you can check if the image is complete before setting the onload event
        if (tempImage.complete) {
            tempImage.onload(); // Manually trigger the onload event
        }

        //SCROLL TO THE RESPONSE MESSAGE
        // Scroll smoothly to the target element
        if (targetElement) {
        targetElement.scrollIntoView({ behavior: "smooth" });
        }
        //

    }
});