import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000

def main():
   if len(sys.argv) != 2:
       sys.exit("Usage: python pagerank.py corpus")
   corpus = crawl(sys.argv[1])
   ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
   print(f"PageRank Results from Sampling (n = {SAMPLES})")
   for page in sorted(ranks):
       print(f"  {page}: {ranks[page]:.4f}")
   ranks = iterate_pagerank(corpus, DAMPING)
   print(f"PageRank Results from Iteration")
   for page in sorted(ranks):
       print(f"  {page}: {ranks[page]:.4f}")
   dict_to_matrix(corpus)

def crawl(directory):
   """
   Parse a directory of HTML pages and check for links to other pages.
   Return a dictionary where each key is a page, and values are
   a list of all other pages in the corpus that are linked to by the page.
   """
   pages = dict()

   # Extract all links from HTML files
   for filename in os.listdir(directory):
       if not filename.endswith(".html"):
           continue
       with open(os.path.join(directory, filename)) as f:
           contents = f.read()
           links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
           pages[filename] = set(links) - {filename}

   # Only include links to other pages in the corpus
   for filename in pages:
       pages[filename] = set(
           link for link in pages[filename]
           if link in pages
       )
   return pages

def transition_model(corpus, page, damping_factor):
  
   transition = dict()

   for dest in corpus:
       transition[dest] = 0
       if dest in corpus[page]:
           transition[dest] += (1 - damping_factor)/len(corpus) + damping_factor*(1/len(corpus[page]))
       else:
           transition[dest] += (1 - damping_factor)/len(corpus)

   return transition
   """
   Return a probability distribution over which page to visit next,
   given a current page.


   With probability `damping_factor`, choose a link at random
   linked to by `page`. With probability `1 - damping_factor`, choose
   a link at random chosen from all pages in the corpus.
   """

def sample_pagerank(corpus, damping_factor, n):
  
   # choose random page
     
   random_page = random.choice(list(corpus.keys()))
   current_page = random_page
  
   # inicialize page values
   valores = dict()
   for page in corpus:
       valores[page] = 0


   for i in range(n):
       # gets the transition probability vector
       transition = transition_model(corpus, current_page, damping_factor)
      
       # gets the pages and weights
       pages = list(transition.keys())
       weights = list(transition.values())


       # calculate next page randomly based on weights
       new_page = random.choices(pages, weights=weights, k=1)[0]


       current_page = new_page
      
       for page in valores:
           if page == new_page:
               valores[page] += 1
  
   for page in valores:
       valores[page] = valores[page]/n


   return valores
   """
   Return PageRank values for each page by sampling `n` pages
   according to transition model, starting with a page at random.


   Return a dictionary where keys are page names, and values are
   their estimated PageRank value (a value between 0 and 1). All
   PageRank values should sum to 1.
   """


def dict_to_matrix(corpus):
   n = len(corpus)
   matrix = np.zeros((n, n))

   # debbuging
   #from pprint import pprint
   #pprint(corpus)

   isolated = 1
   for j, page1 in enumerate(corpus.keys()):
       for i, page2 in enumerate(corpus.keys()):
           if page2 in corpus[page1]:
               matrix[i][j] = 1
               isolated = 0
               #print(f"{page2} esta em {page1}")
       if isolated == 1:
           for i2 in range(n):
               matrix[i2][j] = 1
       isolated = 1

   return matrix

def iterate_pagerank(corpus, damping_factor):
   # Linking matrix
   L = dict_to_matrix(corpus)
   n = len(corpus)
   # Mij matriz (outbound link number)
   M = np.zeros((n, n))
   # Google matrix
   G = np.zeros((n, n))

   #print("Page order: ")
   #for page in corpus.keys():
   #    print(page)
   #print("Matriz L: ", L)

   for i, page in enumerate(corpus):
       if len(corpus[page]) == 0:
           M[i][i] = 1/n
       else:
           M[i][i] = 1/len(corpus[page])

   p = np.full(n, 1/n)
   new_p = np.zeros(n)

   #print("VETOR INICIAL: ", n)
   #print("MATRIZ L: ", L)
   #print("MATRIZ M: ", M)

   itr = 0
   while True:
       end = 1
       G = (1-damping_factor)/n + damping_factor*(L @ M)
       #print("MATRIZ N @ M-1: ", L @ M)
       #print("MATRIZ G: ", G)

       new_p = G @ p
       for pos, value in enumerate(p):
           if abs(value - new_p[pos]) > 0.001:
               end = 0

       p = new_p
      
       itr += 1
       if end == 1:
           break

   print("Numero de iteracoes: ", itr)
  
   dict_p = dict()
   for pos, page in enumerate(corpus.keys()):
       dict_p[page] = new_p[pos]

   return dict_p
   """
   Return PageRank values for each page by iteratively updating
   PageRank values until convergence.


   Return a dictionary where keys are page names, and values are
   their estimated PageRank value (a value between 0 and 1). All
   PageRank values should sum to 1.
   """
if __name__ == "__main__":
   main()
