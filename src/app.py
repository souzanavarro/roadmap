import streamlit as st

def main():
    st.title("Welcome to My Streamlit App")
    st.write("This is a simple Streamlit application.")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    user_input = st.sidebar.text_input("Enter some text:")
    
    if user_input:
        st.write(f"You entered: {user_input}")
    
    # Example of a data visualization
    st.header("Data Visualization")
    data = [1, 2, 3, 4, 5]
    st.line_chart(data)

if __name__ == "__main__":
    main()