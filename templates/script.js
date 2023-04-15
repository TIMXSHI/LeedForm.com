// Retrieve the form element and add an event listener for form submission
document.getElementById('inputForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the form from submitting

    // Retrieve the input values from the form
    var grossFloorArea = document.getElementById('grossFloorArea').value;
    var state = document.getElementById('state').value;
    var ownerTypes = document.getElementById('ownerTypes').value;
    var projectTypes = document.getElementById('projectTypes').value;

    // Create a data object to send to the backend
    var data = {
        'grossFloorArea': grossFloorArea,
        'state': state,
        'ownerTypes': ownerTypes,
        'projectTypes': projectTypes
       
    };

    // Send a POST request to the backend with the form data
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(function (response) {
        return response.json();
    })
    .then(function (data) {
        // Update the prediction result element with the predicted value
        document.getElementById('prediction_result').textContent = 'Predicted Cert Level: ' + data.prediction;
    })
    .catch(function (error) {
        console.error('Error:', error);
    });
});