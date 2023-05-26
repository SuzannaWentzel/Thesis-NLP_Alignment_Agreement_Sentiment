from Helpers import read_csv


__datapath__ = './Data/discussion_post_text_date_author_parents_more_than_two_authors_with_more_than_two_posts.csv' #use the unfiltered csv for original data


"""
Get message statistics
"""
def get_message_length_stats(discussions):
    length_of_the_messages = discussions["text"].str.split("\\s+")
    print('Average length of messages:', length_of_the_messages.str.len().mean())
    print('Min number of words: ', length_of_the_messages.str.len().min())
    print('Max number of words: ', length_of_the_messages.str.len().max())

    messages_more_than_512 = discussions[discussions['text'].str.len() > 512]
    print(messages_more_than_512)
    print('Amount of posts larger than 512 words: ', len(messages_more_than_512.index))



df = read_csv(__datapath__)
get_message_length_stats(df)
