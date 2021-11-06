#The main project file, ie-the flask app. It renders the HTML templates 
#and displays the GUI that is seen by the user.

from flask import Flask, render_template,request
import books13
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

#This allows flask to receive information from the form and display appropriate info on the results page
@app.route('/result',methods = ['POST', 'GET'])

#This is the main function to call the books module
def result():
    titleMain=request.form['title']#item id(book)
    valueOption="book"
    selectOption="average_rating"

    #The user selects book get the lists using the title
    if(valueOption=="book"):

        #If user selects rating, make listrating the mainlist
        if selectOption=="average_rating":
            #listmain=books13.new_recommendation(titleMain)
            # Defining a list
            #listmain=["The Story of the Tampa Bay Buccaneers", "The Gene Makeover", "One Man Out", "Robert Adams", "GED Exercise Books"]
            listmain=books13.book_recommendation_engine(titleMain)

        if not listmain:
            return render_template("error.html",error="list",option=valueOption)

        #Otherwise, display the result screen and find the poster of the book
        else:
            #posterlink="https://www.cnet.com/a/img/-qQkzFVyOPEoBRS7K5kKS0GFDvk=/940x0/2020/04/16/7d6d8ed2-e10c-4f91-b2dd-74fae951c6d8/bazaart-edit-app.jpg"
            #=books.findPoster(titleMain,author)
            return render_template("result.html", titleMain=titleMain, listmain=listmain,valueOption=valueOption)
            
       

          
if __name__ == "__main__":
    app.run(debug=True)

