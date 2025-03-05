import { initializeApp } from "firebase/app";
import {getAuth, GoogleAuthProvider } from "firebase/auth"

const firebaseConfig = {
  apiKey: "AIzaSyDBUatCYvlYDQJNfKKfG0po-aivwTNX31A",
  authDomain: "climalung-175.firebaseapp.com",
  projectId: "climalung-175",
  storageBucket: "climalung-175.firebasestorage.app",
  messagingSenderId: "464652370428",
  appId: "1:464652370428:web:a0581de6102752fadc7279",
  measurementId: "G-VG9PN266RS"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

const auth = getAuth(app);  // Firebase Auth
const googleProvider = new GoogleAuthProvider();  // Google Auth Provider

export { auth, googleProvider };

export default app;