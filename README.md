# Physical Activity Hypoglycemia Risk 

Online browser-based tool for providing an objective hourly hypoglycemia risk score for physical activity decision support leveraging the fixed part of a Mixed Effects Random Forest model (Read more about the model: [Modeling risk of hypoglycemia during and following physical activity in people with type 1 diabetes using explainable mixed-effects machine learning](https://pubmed.ncbi.nlm.nih.gov/36803791/) ).

The development of the web tool was done with [Bokeh](https://docs.bokeh.org/en/latest/), a python library for creating interactive visualizations for web browsers. This tool is publicly available and can be accessed from this [link](https://clara.mosqueralopez.com/pahypoglycemiarisk).

## Getting Started (Only for local usage)

### Package Requirements

Install the following dependencies:

* python=3.8.12
* bokeh=2.4.1
* scikit-learn=1.0.2

### Installation

Clone the repository

`git clone https://github.com/vlt-ro/physical_activity_hypoglycemia_risk_py.git`

### Usage

To run the browser-based tool locally:

`bokeh serve GUI_RandomForest`

And open the following link in the browser:

`localhost:5006/GUI_RandomForest`

### License

Distributed under the MIT License. See `LICENSE.txt` for more information.
