# Thesis: Alignment, Agreement and Sentiment in online discussions
Contains the data processing of my Master Thesis

## Running the project
The alignment can be run with ```main.py```, note that the conversational data should be extracted into ```discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_two_posts.csv```.
This data should have the following columns:
- discussion_id: unique identifier of a discussion which a post belongs to
- post_id: unique identifier of a post
- text: text of the post
- creation_date: date of creation of the post
- author_id: unique identifier of author of post
- parent_post_id: parent post_id of post


## Dependencies
This repo needs the following packages (including their dependencies):
- matplotlib
- nltk
- numba
- numpy
- pandas
- stanza
