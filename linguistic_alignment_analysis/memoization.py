# unique_post_ids => discussion.id & post.id
# unique_words => set van alle woorden in alle posts
#               maak set
#               ga door alle unieke posts
#               voeg elk woord in de post aan de set toe
# unique_authors => author ids

"""
Gebruik pandas



a2: matrix (sparse) vam post, uniek woord
b1: matrix (sparse) van auteur, uniek woord
b1_posts: pandas serie met aantal posts per author
b2: matrix (sparse) van auteur, uniek woord
b2_words: pandas serie met aantal woorden per author


voor elke post:
    counter voor woorden in post
    totaal aantal woorden = len (post)
    voor elk uniek woord in post
        a2[post][woord] = counter (woord) / totaal woorden
        b1[post.author][woord] += 1
        b2[post.author][woord] += counter(woord)
opslaan

b1 = b1 / b1_posts
b2 = b2 / b2_words

opslaan

voor elke discussie:
    voor elke post_response:
        voor elke post_initial in de thread:
            voor elk woord in post_response:
                a1_waarde = woord in post_inital (boolean) -> naar int casten
                a2_waarde = a2[post_response][woord]
                b1_waarde = b1[post_response.author][woord]
                b2_waarde = b2[post_response.author][woord]
                scaled berekenen


"""

