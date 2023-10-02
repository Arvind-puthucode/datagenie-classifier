function plotResults(data) {
    const timestamps = data.result.map(entry => entry.point_timestamp);
    const pointValues = data.result.map(entry => entry.point_value);
    const yhatValues = data.result.map(entry => entry.yhat);

    const plotData = [
        {
            type: 'scatter',
            mode: 'lines',
            x: timestamps,
            y: pointValues,
            name: 'Point Value'
        },
        {
            type: 'scatter',
            mode: 'lines',
            x: timestamps,
            y: yhatValues,
            name: 'Yhat'
        }
    ];

    const plotLayout = {
        title: 'Point Value and Yhat',
        xaxis: {
            title: 'Timestamp'
        },
        yaxis: {
            title: 'Value'
        }
    };

    Plotly.newPlot('plot', plotData, plotLayout);
}

function displayModelInfo(data) {
    const modelNameElement = document.getElementById('model-name');
    modelNameElement.textContent = `Model: ${data.model}`;

    const mapeElement = document.getElementById('mape');
    mapeElement.textContent = `MAPE: ${data.mape}`;
}

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function(event) {
            const fileContent = event.target.result;
            const jsonData = JSON.parse(fileContent);

            fetch('http://0.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'  // Specify the content type as JSON
                },
                body: JSON.stringify(jsonData)  // Send JSON data directly as the body
            })
            .then(response => response.json())
            .then(data => {
                console.log('File uploaded successfully');
                console.log('Prediction result:', data);
                plotResults(data);  // Plot the results
                displayModelInfo(data);  // Display model info
            })
            .catch(error => console.error('Error:', error));
        };

        reader.readAsText(file);
    } else {
        alert('Please select a file.');
    }
}
