<p align="center"><img width="70%" src="docs/images/caspr-logo.png" /></p>

<!-- # AI.Models.CASPR -->
**CASPR is a transformer-based framework for deep learning from sequential data in tabular format, most common in business applications.**

<p align="justify">
Tasks critical to enterprise profitability, such as customer churn prediction, fraudulent account detection or customer lifetime value estimation, are often tackled by models trained on features engineered from customer data in tabular format. Application-specific feature engineering however adds development, operationalization and maintenance costs over time. Recent advances in representation learning present an opportunity to simplify and generalize feature engineering across applications.

With **CASPR**  we propose a novel approach to encode sequential data in tabular format (e.g., customer transactions, purchase history and other interactions) into a generic representation of a subject's (e.g., customer's) association with the business. We evaluate these embeddings as features to train multiple models spanning a variety of applications (see: [paper](https://arxiv.org/abs/2211.09174)). CASPR, Customer Activity Sequence-based Prediction and Representation, applies transformer architecture to encode activity sequences to improve model performance and avoid bespoke feature engineering across applications. Our experiments at scale validate CASPR for both small and large enterprise applications. 
</p>

<!-- - **Representation**      (TODO: in 2 sentences WHY and HOW on CASPR embeddings, RFM)

- **Pre-Training**        (TODO: few words on self-supervised training, platforms supported, pointers to modules)

- **Inference**           (TODO: few words on inference at scale, platforms supported, pointers to modules) -->

## Getting Started & Resources

* **CASPR: Customer Activity Sequence-based Prediction and Representation** (NeurIPS 2022, New Orleans: Tabular Representation Learning)
   - [paper](https://arxiv.org/abs/2211.09174)
   - [poster](https://github.com/microsoft/CASPR/docs/images/caspr-poster.png)

* **Build**

   - pre-requisites:  ```python==3.9, setuptools```
   - building the wheel:  ```python setup.py build bdist_wheel```

* **Installation**

   ```
   (now)
   pip install .\dist\AI.Models.CASPR-<ver>.whl[<optional-env-modifier>]

   (future)
   pip install AI.Models.CASPR[<optional-env-modifier>]
   ```

   use any of below modifiers, to customize the installation for target system / usecase:
   ```
    horovod     - for distributed training and inference on Horovod
    databricks  - for distributed training and inference on Databricks
    aml         - for (distributed) training and inference on Azure ML
    hdi         - for execution on Azure HD Insights
    xai         - to enable explainability
    test        - for extended test execution
    dev         - for development purposes only
   ```
* **Examples**
  
   (TODO: can we point to a well commented one of our examples w/ or w/o data?)

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports please file a [GitHub Issue](https://github.com/Microsoft/CASPR/issues).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).

---