#!/bin/bash

# command that generated the headers.txt file
# jq < data/processed/segmentation_853/raw_mapping.json|jq -r '.[]' | parallel -j1 "cat {}/*.nrrd | awk 'stop==0 {if (\$0 == \"\") stop=1; print \$0 }'" > data/processed/segmentation_853/headers.txt

cat data/processed/segmentation_853/headers.txt | grep -v -e type -e dimension -e sizes -e 'space:' -e 'space directions:' -e 'space origin:' -e kinds -e endian -e encoding -e NRRD0004 -e '# Complete NRRD file format specification at:' -e '# http://teem.sourceforge.net/nrrd/format.html' | awk '$0 != "" {print $0}'

