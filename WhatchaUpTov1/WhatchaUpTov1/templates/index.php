<head>
  <title>HALLO</title>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>

</head>
<body>
  <img src="img/ETH-Logo.jpg"><br>
  Welcome to the entity linking project.
  <br>

<div id="welcom"></div>

<div id="fb-root"></div>
<script src="//connect.facebook.net/en_US/all.js"></script>
<script>

  var session_id;

  FB.init({
    appId  : '731879556882876',
    status : true, // check login status
    cookie : true, // enable cookies to allow the server to access the session
    xfbml  : true, // parse XFBML
    channelUrl : 'http://n.eth.ch/~hmaria/', // channel.html file
    oauth  : true // enable OAuth 2.0
  });
  FB.login(function(response) {
   if (response.authResponse) {
     console.log('Welcome!  Fetching your information.... ');
     FB.api('/me', function(info) {
            logged_in(response, info);
       });    

     FB.Event.subscribe('auth.statusChange', function(response) {
          window.location.reload();
          console.log("EVENT SUBSCRIBE GOT FIRED")
          //window.location.refresh();
        });

   } else {
     console.log('User cancelled login or did not fully authorize.');
   }
 }, {scope: 'public_profile,user_likes,user_hometown,user_location,user_status,user_website,user_work_history,user_education_history'});

 FB.getLoginStatus(function(response) {
  if (response.status === 'connected') {
     session_id = JSON.stringify(response.authResponse.userID);
     var userInfo  =   document.getElementById('user-info');
     console.log('You are connected');
   // the user is logged in and has authenticated your
   // app, and response.authResponse supplies
   // the user's ID, a valid access token, a signed
   // request, and the time the access token 

   // and signed request each expire
    userInfo.innerHTML                             =  'Hello.'
                    document.getElementById('other').style.display = "block";
    
    // Pass user's name:
    FB.api('/me', function(info) {
            logged_in(response, info);
            //session_id = JSON.stringify(info.id);
            var userid = { 'userid': JSON.stringify(info.id) };
    submitInfo(userid)
            var user_name = { 'username': JSON.stringify(info.name) };
    submitInfo(user_name)

       });

    // Pass facebook likes:
    FB.api('/me/likes?limit=5000', function(thing) {
                //console.log('look ' + thing.data.length); //just try to get the length to check wheter it works 
                // I want to add name and category into variables here
                var likes_data = [];
                var likes_categories =[];

                for (i = 0; i < thing.data.length; i++){
                    //console.log('like: ' + thing.data[i].name);
                    likes_data.push(thing.data[i].name);
                    likes_categories.push(thing.data[i].category);
                    }

                var likes_dict = { 'likes': JSON.stringify(likes_data) };
                submitInfo(likes_dict)
                var likes_dict = { 'likes_categories': JSON.stringify(likes_categories) };
                submitInfo(likes_dict)

            });

    // Pass access token:
    var access_token = {'token': JSON.stringify(response.authResponse.accessToken)}
    submitInfo(access_token);

    // Pass user location:
    FB.api('/me', function(user_object) {
            var local = { 'location': JSON.stringify(user_object.location.name) };
    submitInfo(local)
            var home_town = { 'hometown': JSON.stringify(user_object.hometown.name) };
    submitInfo(home_town)

       });

    // Pass other stuff:

 } else if (response.status === 'not_authorized') {
    // the user is logged in to Facebook, 
    // but has not authenticated your app
  } else {
    // the user isn't logged in to Facebook.
  }
});


    function logged_in(response, info){
         if (response.authResponse) {
                    var userInfo  =   document.getElementById('user-info');
                    var accessToken                                 =   response.authResponse.accessToken;
                    
                    //userInfo.innerHTML                             =  info.name 
                    //                                                + "<br /> Your Access Token: " + accessToken ;


                    //document.getElementById('other').style.display = "block";
                    var display_handles = document.getElementById('other');
                    display_handles.innerHTML = '<form id="UserHandles" name="UserHandles">' +

                    'Twitter handle:<input type="text" id="twitterhandle" name=twitterhandle"><br> Foursquare handle:<input type="text" id="four_handle" name="four_handle"><br> Instagram handle:<input type="text" id="insta_handle" name="insta_handle"><br> LinkedIn public profile:<input type="text" id="linkedin_handle" name="linkedin_handle"><br>' +

                    '<input id="submit" type="button" value="Submit"></form>';

                    //$.post('info.php', $('#UserHandles').serialize())

                    /*$("UserHandles").on('submit', function(e){
                    alert("HEY");
                    var twitterhandle = document.forms["handles"]["twitterhandle"].value;
                    console.log(twitterhandle);

                    $.ajax({
                                type: "POST",
                                url: 'info.php',
                                data: { 'twitterhandle': twitterhandle},
                                success: function(result,status,xhr){
                                    console.log(status, result);
                                }
                        });
                    e.preventDefault(); return false });*/
                    $(document).ready(function() {
                        $("#submit").click(function() {
                        var tw = $("#twitterhandle").val();
                        var fs = $("#four_handle").val();
                        var ig = $("#insta_handle").val();
                        var li = $("#linkedin_handle").val();
                        //if (name == '' || email == '' || contact == '' || msg == '') {
                        //alert("Insertion Failed Some Fields are Blank....!!");
                        //} else {
                        // Returns successful data submission message when the entered information is stored in database.
                        $.post("info.php", {
                        twitter: tw,
                        foursquare: fs,
                        instagram: ig,
                        linkedin: li,
                        session_id: session_id
                        }, function(data) {
                        //alert(data);
                        $('#UserHandles')[0].reset(); // To reset form fields
                        });
                        //}
                        });
                        });

                }
            }

    function submitInfo(var1, async_flag)
    {
     async_flag = typeof async_flag !== 'undefined' ? async_flag : true;
     console.log(var1);
     console.log(session_id);
     var1["session_id"]= session_id;
     $.ajax({
                    type: "POST",
                    url: 'info.php',
                    data: var1,
                    async: async_flag,
                    success: function(result,status,xhr){
                        console.log(status, result);
                    }
            });
    }


</script>

        <div id="other" style="display:none"></div>
        <div id="user-info"></div>

</body>
