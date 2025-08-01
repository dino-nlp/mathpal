# MathPal 🤖

 An AI-powered math buddy to help students transition from 5th to 6th grade.

## GUIDE

- Install libs: `make install`
- Start infra: `make local-start`
- Stop infra: `make local-stop`
- Crawl test data: `make local-test-crawler`
- Crawl full data: `make local-ingest-data`
- Test retriever: `make local-test-retriever`

## **IMPORTANT NOTE:**

- For MongoDB to work with multiple replicas (as we use it in our Docker setup) on macOS or Linux systems, you have to add the following lines of code to `/etc/hosts`:

  - 127.0.0.1       mongo1
  - 127.0.0.1       mongo2 
  - 127.0.0.1       mongo3

-  Qdrant UI: `localhost:6333/dashboard`