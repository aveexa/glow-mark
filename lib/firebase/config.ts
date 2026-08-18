import { initializeApp, getApps, FirebaseApp } from "firebase/app";
import { getAuth, Auth } from "firebase/auth";
import { getFirestore, Firestore } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAkwKIuyU8qGP4V4-Y4Ic2cCEX1nohtxss",
  authDomain: "glowmark-ai.firebaseapp.com",
  projectId: "glowmark-ai",
  storageBucket: "glowmark-ai.firebasestorage.app",
  messagingSenderId: "538448190596",
  appId: "1:538448190596:web:71dbd37e86df156fc836d1",
  measurementId: "G-M43TLFVSQS"
};

// Initialize Firebase
let app: FirebaseApp;
if (!getApps().length) {
  app = initializeApp(firebaseConfig);
} else {
  app = getApps()[0];
}

// Initialize Firebase services
export const auth: Auth = getAuth(app);
export const db: Firestore = getFirestore(app);

export default app;
