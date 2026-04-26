# Sample Datasets

There are loads of example datasets available on OpenExchange and easy to install with InterSystems Package Manager. IPM is already installed in the build process, so all you need to do is run: 

```bash
docker-compose exec -it iris iris session iris
```

```objectscript
set $NAMESPACE = "IRISAPP" // Current Working Namespace

// install example package
zpm "install dataset-health"
```

Example datasets on Open Exchange: 


| Name                   | Description                                                                                                                                | Install                                                                                                                | Link                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Health Dataset         | 10 datasets with real data covering major diseases                                                                                         | `zpm "install dataset-health"`                                                                                         | [Package](https://openexchange.intersystems.com/package/Health-Dataset)         |
| Medical Datasets       | 12 datasets with 43 tables total.                                                                                                          | `zpm "install dataset-medical"` , Then run: <br>`do ##class(dc.data.medical.utility).ImportDS(999)` to install all<br> | [Package](https://openexchange.intersystems.com/package/Medical-Datasets)       |
| Finance Dataset        | Table of financial transactions                                                                                                            | `zpm "install dataset-finance`                                                                                         | [Package](https://openexchange.intersystems.com/package/dataset-finance)        |
| SyntheticDataGen       | Tool for generating 4 different configurable synthetic datasets (5-10 linked tables) covering finance, supplychain, retail and a themepark | `zpm "install iris-synthetic-data-gen"` and follow instructions in link                                                                               |       [Package](https://openexchange.intersystems.com/package/iris-synthetic-data-gen)                                                                          |
| iris-dataset-countries | Countries by longitude latitude and region                                                                                                 | `zpm "install dataset-countries"`                                                                                      | [Package](https://openexchange.intersystems.com/package/iris-dataset-countries) |
| exchange-rate-cbrf     | Exchange rates from Central bank of russian federation                                                                                     | `zpm "install exchange-rate-cbrf"`                                                                                     | [Package](https://openexchange.intersystems.com/package/exchange-rate-cbrf)     |



Or choose to a dataset from another source with the help of one of these packages: 

| Name                          | Description                              | Install                                       | Link                                                                                   |
| ----------------------------- | ---------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| csvgen                        | Generate IRIS table from CSV             | `zpm "install csvgen"`                        | [Package](https://openexchange.intersystems.com/package/csvgen)                        |
| iris-kaggle-socrata-generator | Download datastes from kaggle or Socrata | `zpm "install iris-kaggle-socrata-generator"` | [Package](https://openexchange.intersystems.com/package/iris-kaggle-socrata-generator) |

Other links: 

- [Find Open Datasets and Machine Learning Projects | Kaggle](https://www.kaggle.com/datasets)
- [Open Data Network](https://www.opendatanetwork.com/)
- [Open Data on AWS](https://aws.amazon.com/opendata/)
