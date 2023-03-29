function submitCheck() {
    var form = document.getElementById("main-form");
    var x = document.getElementById("b-camel").parentElement.id;
    form.append('<input type="hidden" name="b-camel" value="' + x + '" />');
    return true;
}

function getBlueCamelLocation() {
    return document.getElementById("b-camel").parentElement.id;
}