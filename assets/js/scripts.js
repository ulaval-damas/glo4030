var collapsibles = document.getElementsByClassName("collapsible");

for (var i = 0; i < collapsibles.length; i++) {
    var collapsibleList = collapsibles[i].children;
    for (var j = 0; j < collapsibleList.length; j++) {
        var collapsibleElement = collapsibleList[j];
        console.log(collapsibleElement);
        collapsibleElement.firstElementChild.onclick = function(){
            this.classList.toggle("active");

            var content = this.nextElementSibling;
            while (content){
                if (content.style.display !== "block") {
                    content.style.display = "block";
                } else {
                    content.style.display = "none";
                }
                content = content.nextElementSibling;
            }
        };
    }
}
