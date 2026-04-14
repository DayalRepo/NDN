# Deep Learning Based Detection of Interest Flooding Attacks in Named Data Networking using ndnSIM

## 🔵 PART 1 — PROJECT OVERVIEW

### What is Named Data Networking (NDN)?

Named Data Networking (NDN) is a future internet architecture that shifts the focus from IP addresses to data names. Instead of communicating with specific hosts, consumers request data by its hierarchical name using an "Interest" packet. The network routes this Interest to data producers, and the requested "Data" packet flows back along the reverse path.

### What is an Interest Flooding Attack (IFA)?

An Interest Flooding Attack (IFA) is a Distributed Denial of Service (DDoS) attack specific to NDN. Attackers send a massive volume of Interest packets for non-existent or dynamically generated data names. Since the data does not exist, the network's Pending Interest Tables (PIT) keep these states active until they time out.

### Why PIT Exhaustion occurs?

Every forwarded Interest leaves a trail in the PIT of a router so the corresponding Data packet can find its way back. In an IFA, because the Data is never returned, the PIT entries persist until their timeout expires. The onslaught of malicious Interests quickly exhausts the PIT memory, preventing legitimate Interests from being processed.

### Why Deep Learning (CNN) is suitable?

Traditional threshold-based methods fail against sophisticated IFA variants. A 1D Convolutional Neural Network (CNN) can effectively capture the temporal and spatial patterns of network traffic (Interest rates, Satisfaction ratios, Timeouts) over sequential time windows. CNNs extract automated features from raw traffic logs, providing high accuracy with minimal manual feature engineering.

### System Objectives

1. Simulate realistic NDN traffic and IFA vulnerabilities using ndnSIM.
2. Generate comprehensive traffic logs with normal and varied attack profiles.
3. Build and train a 1D CNN model to accurately classify network states.
4. Develop a Flask-based REST API to serve the trained model.
5. Create a dynamic, interactive web dashboard for real-time traffic monitoring and attack detection alerts.

### Expected Outputs

1. A realistic ndnSIM traffic dataset containing normal and attack scenarios.
2. A high-accuracy CNN model for IFA detection.
3. A functional web dashboard correctly classifying uploaded traffic logs and displaying metrics.

---

## 🔵 PART 2 — PROJECT ARCHITECTURE

### System Architecture Diagram

```text
[ndnSIM Simulation] --> (Normal/Attack Traffic Logs)
          |
          v
   [Data Extraction] --> (CSV Dataset)
          |
          v
[Data Preprocessing] --> (Feature Engineering & Normalization)
          |
          v
   [CNN 1D Model] <--- (Training & Validation) --> [Saved Model: ndn_cnn_model.h5]
          |
          v
   [Flask Backend] <--- (API Route: /predict)
          |
          v
 [Web Dashboard UI] --> (Charts, Metrics, Alert Banners)
```

### Module Explainations:

1. **ndnSIM Simulation**: Configured to simulate NDN network topologies, legitimate consumer activity, and malicious attacker nodes (Interest Flooding, Slow IFA, Cache Pollution).
2. **Data Extraction & CSV Generation**: Network trace files are aggregated into a structured CSV format with specific metrics per node over time.
3. **Preprocessing**: The CSV data is cleaned. Derived features like `interest_rate`, `satisfaction_ratio`, and `timeout_ratio` are computed to frame the dataset as a time-series problem.
4. **CNN Model**: A 1D Convolutional Neural Network processes the time-windowed traffic features to learn patterns distinguishing normal from malicious states.
5. **Flask API**: Serves as the middle layer, loading the trained model and providing an endpoint to receive new traffic data (CSV) and return predictions in JSON format.
6. **Web Dashboard**: The frontend built with HTML/CSS/JS and Chart.js. It calls the Flask API, parsing the JSON to display dynamic graphs and alerting the user if an attack is detected.

---

## 🔵 PART 8 — PROJECT FOLDER STRUCTURE

```text
C437 Flooding Attack/
│
├── README.md                      # Project documentation (Overview, Architecture, Setup)
├── requirements.txt               # Python dependencies
│
├── ndnsim/                        # C++ code for ndnSIM
│   └── attack_simulation.cc       # Example script for attacks
│
├── dataset/                       # Data processing & dummy generation
│   └── generate_dummy_csv.py      # Script to generate a sample dataset
│
├── notebook/                      # Machine learning
│   └── cnn_model_training.ipynb   # Jupyter Notebook for preprocessing & CNN
│
├── model/                         # Saved trained models
│   └── ndn_cnn_model.h5           # (Generated after training)
│
└── flask_app/                     # Web Dashboard
    ├── app.py                     # Flask server and API
    ├── templates/
    │   └── index.html             # Frontend layout
    └── static/
        ├── style.css              # Custom styling
        └── script.js              # API calls and Chart.js logic
```

---

## 🔵 PART 9 — INSTALLATION GUIDE

### 1. Python & Dependencies Install

Ensure Python 3.8+ is installed. Then, install the required packages using the terminal:

```bash
cd "C437 Flooding Attack"
pip install -r requirements.txt
```

### 2. ndnSIM Setup (If you want to run simulations)

ndnSIM requires a Linux environment (Ubuntu recommended). Follow the official ndnSIM documentation to install ns-3 and ndnSIM requirements. You can compile the code provided in the `ndnsim/` folder within the `scratch/` directory of your ns-3 installation.

### 3. Generate Dummy Dataset

If you don't have a real ndnSIM dataset, generate a dummy one for testing:

```bash
python dataset/generate_dummy_csv.py
```

This will create `dataset/ndn_traffic.csv`.

### 4. How to Train the Model

1. Open the Jupyter Notebook:

```bash
jupyter notebook notebook/cnn_model_training.ipynb
```

2. Run all the cells in the notebook. It will process the CSV, train the 1D CNN model, and save it automatically to `model/ndn_cnn_model.h5`.

### 5. How to Run the Server

1. Navigate to the `flask_app` directory:

```bash
cd flask_app
```

2. Run the application:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/`.

### 6. How to Test Detection

1. Open a web browser and go to `http://127.0.0.1:5000/`.
2. Click "Choose File" and upload the `dataset/ndn_traffic.csv` (or an excerpt of it).
3. Click "Analyze Traffic".
4. The dashboard will dynamically plot the interest parameters and show a status banner (Normal or Attack Detected with severity).

---

## 🔵 PART 10 — VIVA QUESTIONS AND ANSWERS & IMPROVEMENTS

### Possible Viva Questions

**Q1: What is the primary vulnerability in NDN that leads to Interest Flooding Attacks?**
_Answer:_ The Stateful forwarding plane. NDN routers maintain a Pending Interest Table (PIT) to route data back to consumers. If an attacker requests non-existent data, the PIT entry stays active until a timeout occurs, exhausting memory.

**Q2: Why did you choose a 1D CNN instead of standard Machine Learning algorithms like SVM or Random Forest?**
_Answer:_ Network traffic exhibits sequential, temporal dependencies. A 1D CNN efficiently slides over time domains to extract features directly from raw metrics (like interest rate over time) and recognizes local spatial-temporal patterns indicative of varying attack intensities.

**Q3: How does Cache Pollution differ from an Interest Flooding Attack?**
_Answer:_ IFA targets router Memory (PIT) by requesting non-existent data. Cache Pollution targets the Content Store (CS) by requesting unpopular data to fill the cache, evicting popular data and causing cache misses for legitimate users.

**Q4: Explain the 'satisfaction_ratio' feature.**
_Answer:_ It is the ratio of `InSatisfiedInterests` to `InInterests`. Under normal conditions, this is close to 1. During an IFA (where data doesn't exist), this ratio drops dramatically.

### Suggestions for Improving Accuracy

1. **Long Short-Term Memory (LSTM) Networks:** Using LSTMs or a hybrid CNN-LSTM could capture longer-term temporal dependencies in the traffic logs.
2. **Larger Topology Datasets:** Simulating larger and more complex topologies (like DFN or AT&T topologies) with varying link delays.
3. **Real-time Feature Extraction:** Migrating from CSV batch uploads to a Kafka-based real-time streaming pipeline.
