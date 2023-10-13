sed -n '/^START=A$/,/^END$/p' data

# Performing blasts with the public rest API of NCBI BLAST

before submitting a query, a FASTA file needs to be prepared, with the 
gene sequence(s) of interest. This is a plain text file, which lists
genes according to this sequence.

```
>sequence_1
GTTTTGTTTATTACTCTGTCATAACACAGATATTTATTACTGCTTTGGAGCGTA
```

Now a request can be submitted. The options `PROGRAM=blastn` for referencing nucleotide
sequences, `DATABASE=Danio rerio` for accessing the Danio rerio reference genome
are given and the fasta sequence is past. 

The database names are difficult. I am trying to use the ones from the `.asn`
files available from ncbi results search strategy download
<https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=GetSaved&RECENT_RESULTS=on>


```bash
curl -X POST -d "CMD=Put" -d "PROGRAM=blastn" -d "MEGABLAST=on" -d "DATABASE=GPIPE/7955/106/ref_top_level" --data-urlencode "QUERY@data/processed_data/sequence.fasta" "https://www.ncbi.nlm.nih.gov/blast/Blast.cgi" > work/blast_request.html
```
The output html document 
is forwarded to a temporary document which contains the relevant info:

```
 <!--QBlastInfoBegin
    RID = SYZDXEWK014  
    RTOE = 31  
 QBlastInfoEnd
 -->
```

These results can be extracted by using `sed`

```bash
RID=`sed -n '/^<!--QBlastInfoBegin$/,/^QBlastInfoEnd$/p' work/blast_request.html | head -n 2 | tail -n 1 | tr -d '[:space:]' | tr -d 'RID='`
RTOE=`sed -n '/^<!--QBlastInfoBegin$/,/^QBlastInfoEnd$/p' work/blast_request.html | head -n 3 | tail -n 1 | tr -d '[:space:]' | tr -d 'RTOE='`
```

query if results are done. This also outputs a html file :/

```bash
curl -d "CMD=Get" -d "FORMAT_OBJECT=SearchInfo" -d "RID=$RID" "https://www.ncbi.nlm.nih.gov/blast/Blast.cgi" > work/blast_query_status.html
```

Then after the RTOE is over I can get the results 

```bash
curl -d "CMD=Get" -d "FORMAT_TYPE=XML" -d "RID=$RID" "https://www.ncbi.nlm.nih.gov/blast/Blast.cgi" > work/blast_query.xml
```

from there via the search ensembl:
`18:20613331-20613384`, I could get the information. 18 is the chromosome and 
the range afterwards is the location in the chromosome